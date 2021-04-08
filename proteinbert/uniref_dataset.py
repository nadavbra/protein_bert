import re
import gzip
import json
from collections import Counter
import sqlite3

import numpy as np
import pandas as pd
import h5py
from lxml import etree
from pyfaidx import Faidx

from .shared_utils.util import log, to_chunks

class UnirefToSqliteParser:

    def __init__(self, uniref_xml_gz_file_path, go_annotations_meta, sqlite_file_path, verbose = True, log_progress_every = 1000, \
            chunk_size = 100000):
        
        self.uniref_xml_gz_file_path = uniref_xml_gz_file_path
        self.go_annotations_meta = go_annotations_meta
        self.sqlite_conn = sqlite3.connect(sqlite_file_path)
        self.verbose = verbose
        self.log_progress_every = log_progress_every
        self.chunk_size = chunk_size
        
        self._process_go_annotations_meta()
        
        self.go_index_record_counter = Counter()
        self.unrecognized_go_annotations = Counter()
        self.n_records_with_any_go_annotation = 0
                
        self._chunk_indices = []
        self._chunk_records = []
        
    def parse(self):
        
        with gzip.open(self.uniref_xml_gz_file_path, 'rb') as f:
            context = etree.iterparse(f, tag = UnirefToSqliteParser._NAMESPACE_PREFIX + 'entry', events = ('end',))
            _etree_fast_iter(context, self._process_entry)
        
        if len(self._chunk_records) > 0:
            self._save_current_chunk()
            
        if self.verbose:
            log('Ignored the following unrecognized GO annotations: %s' % self.unrecognized_go_annotations)
            log('Parsed %d records with any GO annotation.' % self.n_records_with_any_go_annotation)
            
        go_id_record_counter = pd.Series(self.go_index_record_counter)
        go_id_record_counter.index = [self.go_index_to_id[index] for index in go_id_record_counter.index]

        self.go_annotations_meta['count'] = go_id_record_counter.reindex(self.go_annotations_meta.index).fillna(0)
        self.go_annotations_meta['freq'] = self.go_annotations_meta['count'] / self.n_records_with_any_go_annotation
              
        if self.verbose:
            log('Done.')
            
    def _process_go_annotations_meta(self):
        self.go_annotation_to_all_ancestors = self.go_annotations_meta['all_ancestors'].to_dict()
        self.go_id_to_index = self.go_annotations_meta['index'].to_dict()
        self.go_index_to_id = self.go_annotations_meta.reset_index().set_index('index')['id'].to_dict()
        
    def _process_entry(self, i, event, entry):
            
        if self.verbose and i % self.log_progress_every == 0:
            log(i, end = '\r')

        repr_member, = entry.xpath(r'uniprot:representativeMember', namespaces = UnirefToSqliteParser._NAMESPACES)
        db_ref, = repr_member.xpath(r'uniprot:dbReference', namespaces = UnirefToSqliteParser._NAMESPACES)
        protein_name = db_ref.attrib['id']

        try:
            taxonomy_element, = db_ref.xpath(r'uniprot:property[@type="NCBI taxonomy"]', namespaces = UnirefToSqliteParser._NAMESPACES)
            tax_id = int(taxonomy_element.attrib['value'])
        except:
            tax_id = np.nan

        extracted_go_annotations = {category: UnirefToSqliteParser._extract_go_category(entry, category) for category in UnirefToSqliteParser._GO_ANNOTATION_CATEGORIES}
        
        self._chunk_indices.append(i)
        self._chunk_records.append((tax_id, protein_name, extracted_go_annotations))
        
        if len(self._chunk_records) >= self.chunk_size:
            self._save_current_chunk()

    def _save_current_chunk(self):
        
        chunk_records_df = pd.DataFrame(self._chunk_records, columns = ['tax_id', 'uniprot_name', 'go_annotations'], index = self._chunk_indices)
        
        chunk_records_df['flat_go_annotations'] = chunk_records_df['go_annotations'].apply(\
                lambda go_annotations: list(sorted(set.union(*map(set, go_annotations.values())))))
        chunk_records_df['n_go_annotations'] = chunk_records_df['flat_go_annotations'].apply(len)
        chunk_records_df['complete_go_annotation_indices'] = chunk_records_df['flat_go_annotations'].apply(self._get_complete_go_annotation_indices)
        chunk_records_df['n_complete_go_annotations'] = chunk_records_df['complete_go_annotation_indices'].apply(len)
        self.n_records_with_any_go_annotation += (chunk_records_df['n_complete_go_annotations'] > 0).sum()
        
        for complete_go_annotation_indices in chunk_records_df['complete_go_annotation_indices']:
            self.go_index_record_counter.update(complete_go_annotation_indices)

        chunk_records_df['go_annotations'] = chunk_records_df['go_annotations'].apply(json.dumps)
        chunk_records_df['flat_go_annotations'] = chunk_records_df['flat_go_annotations'].apply(json.dumps)
        chunk_records_df['complete_go_annotation_indices'] = chunk_records_df['complete_go_annotation_indices'].apply(json.dumps)
        chunk_records_df.to_sql('protein_annotations', self.sqlite_conn, if_exists = 'append')
        
        self._chunk_indices = []
        self._chunk_records = []
        
    def _get_complete_go_annotation_indices(self, go_annotations):
        complete_go_annotations = self._get_complete_go_annotations(go_annotations)
        return list(sorted(filter(None, map(self.go_id_to_index.get, go_annotations))))

    def _get_complete_go_annotations(self, go_annotations):
        return set.union(set(), *[self._get_go_annotation_all_ancestors(annotation) for annotation in go_annotations])
    
    def _get_go_annotation_all_ancestors(self, annotation):
        if annotation in self.go_annotation_to_all_ancestors:
            return self.go_annotation_to_all_ancestors[annotation]
        else:
            self.unrecognized_go_annotations[annotation] += 1
            return set()
           
    @staticmethod
    def _extract_go_category(entry, category):
        return list({property_element.attrib['value'] for property_element in entry.xpath(r'uniprot:property[@type="%s"]' % \
                category, namespaces = UnirefToSqliteParser._NAMESPACES)})
            
    _NAMESPACE_PREFIX = r'{http://uniprot.org/uniref}'
    _NAMESPACES = {'uniprot': r'http://uniprot.org/uniref'}

    _GO_ANNOTATION_CATEGORIES = [
        'GO Molecular Function',
        'GO Biological Process',
        'GO Cellular Component',
    ]

def parse_go_annotations_meta(meta_file_path):

    ALL_FIELDS = ['id', 'name', 'namespace', 'def', 'is_a', 'synonym', 'alt_id', 'subset', 'is_obsolete', 'xref', \
            'relationship', 'intersection_of', 'disjoint_from', 'consider', 'comment', 'replaced_by', 'created_by', \
            'creation_date', 'property_value']
    LIST_FIELDS = {'synonym', 'alt_id', 'subset', 'is_a', 'xref', 'relationship', 'disjoint_from', 'intersection_of', \
            'consider', 'property_value'}
            
    GO_ANNOTATION_PATTERN = re.compile(r'\[Term\]\n((?:\w+\: .*\n?)+)')
    FIELD_LINE_PATTERN = re.compile(r'(\w+)\: (.*)')
    
    with open(meta_file_path, 'r') as f:
        raw_go_meta = f.read()

    go_annotations_meta = []

    for match in GO_ANNOTATION_PATTERN.finditer(raw_go_meta):
        
        raw_go_annotation = match.group(1)
        go_annotation = {field: [] for field in LIST_FIELDS}
        
        for line in raw_go_annotation.splitlines():
            
            (field, value), = FIELD_LINE_PATTERN.findall(line)
            assert field in ALL_FIELDS
            
            if field in LIST_FIELDS:
                go_annotation[field].append(value)
            else:
                assert field not in go_annotation
                go_annotation[field] = value
        
        go_annotations_meta.append(go_annotation)

    go_annotations_meta = pd.DataFrame(go_annotations_meta, columns = ALL_FIELDS)
    go_annotations_meta['is_obsolete'] = go_annotations_meta['is_obsolete'].fillna(False)
    assert go_annotations_meta['id'].is_unique
    go_annotations_meta.set_index('id', drop = True, inplace = True)
    go_annotations_meta.insert(0, 'index', np.arange(len(go_annotations_meta)))
    _add_children_and_parents_to_go_annotations_meta(go_annotations_meta)
    
    return go_annotations_meta
    
def create_h5_dataset(protein_annotations_sqlite_db_file_path, fasta_file_path, go_annotations_meta_csv_file_path, output_h5_file_path, shuffle = True, \
        min_records_to_keep_annotation = 100, records_limit = None, save_chunk_size = 10000, verbose = True, log_progress_every = 10000):
    
    go_annotations_meta = pd.read_csv(go_annotations_meta_csv_file_path, usecols = ['id', 'index', 'count'], index_col = 0)
    annotation_counts = go_annotations_meta['count']
    common_annotation_ids = np.array(sorted(annotation_counts[annotation_counts >= min_records_to_keep_annotation].index))
    original_annotation_index_to_common_annotation_index = {go_annotations_meta.loc[annotation_id, 'index']: i for i, annotation_id in enumerate(common_annotation_ids)}
    
    if verbose:
        log('Will encode the %d most common annotations.' % len(common_annotation_ids))
        
    n_seqs = sum(1 for _ in load_seqs_and_annotations(protein_annotations_sqlite_db_file_path, fasta_file_path, shuffle = False, records_limit = records_limit, \
            verbose = verbose, log_progress_every = log_progress_every))
            
    if verbose:
        log('Will create an h5 dataset of %d final sequences.' % n_seqs)

    with h5py.File(output_h5_file_path, 'w') as h5f:
        
        h5f.create_dataset('included_annotations', data = [annotation.encode('ascii') for annotation in common_annotation_ids], dtype = h5py.string_dtype())
        uniprot_ids = h5f.create_dataset('uniprot_ids', shape = (n_seqs,), dtype = h5py.string_dtype())
        seqs = h5f.create_dataset('seqs', shape = (n_seqs,), dtype = h5py.string_dtype())
        seq_lengths = h5f.create_dataset('seq_lengths', shape = (n_seqs,), dtype = np.int32)
        annotation_masks = h5f.create_dataset('annotation_masks', shape = (n_seqs, len(common_annotation_ids)), dtype = bool)
        
        start_index = 0
        
        for seqs_and_annotations_chunk in to_chunks(load_seqs_and_annotations(protein_annotations_sqlite_db_file_path, fasta_file_path, shuffle = shuffle, \
                records_limit = records_limit, verbose = verbose, log_progress_every = log_progress_every), save_chunk_size):
            
            end_index = start_index + len(seqs_and_annotations_chunk)
            uniprot_id_chunk, seq_chunk, annotation_indices_chunk = map(list, zip(*seqs_and_annotations_chunk))
            
            uniprot_ids[start_index:end_index] = uniprot_id_chunk
            seqs[start_index:end_index] = seq_chunk
            seq_lengths[start_index:end_index] = list(map(len, seq_chunk))
            annotation_masks[start_index:end_index, :] = _encode_annotations_as_a_binary_matrix(annotation_indices_chunk, original_annotation_index_to_common_annotation_index)
            
            start_index = end_index
             
    if verbose:
        log('Done.')

def load_seqs_and_annotations(protein_annotations_sqlite_db_file_path, fasta_file_path, shuffle = True, records_limit = None, verbose = True, \
        log_progress_every = 10000):

    if verbose:
        log('Loading %s records...' % ('all' if records_limit is None else records_limit))
    
    conn = sqlite3.connect(protein_annotations_sqlite_db_file_path)
    raw_proteins_and_annotations = pd.read_sql_query('SELECT uniprot_name, complete_go_annotation_indices FROM protein_annotations' + ('' if records_limit is None else \
            (' LIMIT %d' % records_limit)), conn)
    
    if verbose:
        log('Loaded %d proteins and their GO annotations (%d columns: %s)' % (raw_proteins_and_annotations.shape + (', '.join(raw_proteins_and_annotations.columns),)))
    
    if shuffle:
        raw_proteins_and_annotations = raw_proteins_and_annotations.sample(frac = 1, random_state = 0)

    if verbose:
        log('Loading Faidx (%s)...' % fasta_file_path)
    
    seqs_faidx = Faidx(fasta_file_path)
    
    if verbose:
        log('Finished loading Faidx.')
    
    n_failed = 0

    for i, (_, (uniprot_id, raw_go_annotation_indices)) in enumerate(raw_proteins_and_annotations.iterrows()):
        
        if verbose and i % log_progress_every == 0:
            log('%d/%d' % (i, len(raw_proteins_and_annotations)), end = '\r')
        
        seq_fasta_id = 'UniRef90_%s' % uniprot_id.split('_')[0]
        
        try:
            seq = str(seqs_faidx.fetch(seq_fasta_id, 1, seqs_faidx.index[seq_fasta_id].rlen))
            yield uniprot_id, seq, json.loads(raw_go_annotation_indices)
        except KeyError:
            n_failed += 1
            
    if verbose:
        log('Finished. Failed finding the sequence for %d of %d records.' % (n_failed, len(raw_proteins_and_annotations)))
    
def _add_children_and_parents_to_go_annotations_meta(go_annotations_meta):

    go_annotations_meta['direct_children'] = [set() for _ in range(len(go_annotations_meta))]
    go_annotations_meta['direct_parents'] = [set() for _ in range(len(go_annotations_meta))]

    for go_id, go_annotation in go_annotations_meta.iterrows():
        for raw_is_a in go_annotation['is_a']:
            parent_id, parent_name = raw_is_a.split(' ! ')
            parent_go_annotation = go_annotations_meta.loc[parent_id]
            assert parent_go_annotation['name'] == parent_name
            go_annotation['direct_parents'].add(parent_id)
            parent_go_annotation['direct_children'].add(go_id)
            
    go_annotations_meta['all_ancestors'] = pd.Series(_get_index_to_all_ancestors(\
            go_annotations_meta['direct_children'].to_dict(), \
            go_annotations_meta[~go_annotations_meta['direct_parents'].apply(bool)].index))
    go_annotations_meta['all_offspring'] = pd.Series(_get_index_to_all_ancestors(\
            go_annotations_meta['direct_parents'].to_dict(), \
            go_annotations_meta[~go_annotations_meta['direct_children'].apply(bool)].index))

def _get_index_to_all_ancestors(index_to_direct_children, root_indices):
    
    index_to_all_ancestors = {index: {index} for index in index_to_direct_children.keys()}
    indices_to_scan = set(root_indices)
    
    while indices_to_scan:
        
        scanned_child_indices = set()
        
        for index in indices_to_scan:
            for child_index in index_to_direct_children[index]:
                index_to_all_ancestors[child_index].update(index_to_all_ancestors[index])
                scanned_child_indices.add(child_index)
                
        indices_to_scan = scanned_child_indices
        
    return index_to_all_ancestors
    
def _encode_annotations_as_a_binary_matrix(records_annotations, annotation_to_index):
    
    annotation_masks = np.zeros((len(records_annotations), len(annotation_to_index)), dtype = bool)
    
    for i, record_annotations in enumerate(records_annotations):
        for annotation in record_annotations:
            if annotation in annotation_to_index:
                annotation_masks[i, annotation_to_index[annotation]] = True
            
    return annotation_masks
        
def _etree_fast_iter(context, func, func_args = [], func_kwargs = {}, max_elements = None):
    '''
    Based on: https://stackoverflow.com/questions/12160418/why-is-lxml-etree-iterparse-eating-up-all-my-memory
    http://lxml.de/parsing.html#modifying-the-tree
    Based on Liza Daly's fast_iter
    http://www.ibm.com/developerworks/xml/library/x-hiperfparse/
    See also http://effbot.org/zone/element-iterparse.htm
    '''
    for i, (event, elem) in enumerate(context):
        func(i, event, elem, *func_args, **func_kwargs)
        # It's safe to call clear() here because no descendants will be
        # accessed
        elem.clear()
        # Also eliminate now-empty references from the root node to elem
        for ancestor in elem.xpath('ancestor-or-self::*'):
            while ancestor.getprevious() is not None:
                del ancestor.getparent()[0]
        if max_elements is not None and i >= max_elements - 1:
            break
    del context
