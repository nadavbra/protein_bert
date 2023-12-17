import os
import shutil
from urllib.parse import urlparse
from urllib.request import urlopen

from tensorflow import keras

from . import conv_and_global_attention_model
from .model_generation import load_pretrained_model_from_dump

DEFAULT_LOCAL_MODEL_DUMP_DIR = '~/proteinbert_models'
DEFAULT_LOCAL_MODEL_DUMP_FILE_NAME = 'default.pkl'
DEFAULT_REMOTE_MODEL_DUMP_URL = 'https://media.githubusercontent.com/media/nadavbra/proteinbert_data_files/master/epoch_92400_sample_23500000.pkl'

def load_pretrained_model(local_model_dump_dir = DEFAULT_LOCAL_MODEL_DUMP_DIR, local_model_dump_file_name = DEFAULT_LOCAL_MODEL_DUMP_FILE_NAME, \
        remote_model_dump_url = DEFAULT_REMOTE_MODEL_DUMP_URL, download_model_dump_if_not_exists = True, validate_downloading = True, \
        create_model_function = conv_and_global_attention_model.create_model, create_model_kwargs = {}, optimizer_class = keras.optimizers.Adam, lr = 2e-04, \
        other_optimizer_kwargs = {}, annots_loss_weight = 1, load_optimizer_weights = False):

    local_model_dump_dir = os.path.expanduser(local_model_dump_dir)
    dump_file_path = os.path.join(local_model_dump_dir, local_model_dump_file_name)
    
    if not os.path.exists(dump_file_path) and download_model_dump_if_not_exists:
        
        if validate_downloading:
            
            print(f' Local model dump file {dump_file_path} doesn\'t exist. Will download {remote_model_dump_url} into {local_model_dump_dir}. Please approve or reject this ' + \
                    '(to exit and potentially call the function again with different parameters).')
            
            while True:
                
                user_input = input('Do you approve downloading the file into the specified directory? Please specify "Yes" or "No":')
                
                if user_input.lower() in {'yes', 'y'}:
                    break
                elif user_input.lower() in {'no', 'n'}:
                    raise ValueError('User wished to cancel.')
        
        downloaded_file_name = os.path.basename(urlparse(remote_model_dump_url).path)
        if not os.path.exists(local_model_dump_dir):
            os.mkdir(local_model_dump_dir)
        downloaded_file_path = os.path.join(local_model_dump_dir, downloaded_file_name)
        assert not os.path.exists(downloaded_file_path), 'Cannot download into an already existing file: %s' % downloaded_file_path
        
        with urlopen(remote_model_dump_url) as remote_file, open(downloaded_file_path, 'wb') as local_file:
            shutil.copyfileobj(remote_file, local_file)
        
        print('Downloaded file: %s' % downloaded_file_path)
            
        if downloaded_file_name != local_model_dump_file_name:
            os.symlink(downloaded_file_path, dump_file_path)
            print('Created: %s' % dump_file_path)        
    
    return load_pretrained_model_from_dump(dump_file_path, create_model_function, create_model_kwargs = create_model_kwargs, optimizer_class = optimizer_class, lr = lr, \
            other_optimizer_kwargs = other_optimizer_kwargs, annots_loss_weight = annots_loss_weight, load_optimizer_weights = load_optimizer_weights)
