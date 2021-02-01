import os

def run_script(epochs, optimizer, optimizer_shortform, loss_function, loss_function_shortform, batchsize):
    os.chdir("../")
    cmd0 = '''python3 setup.py install'''
    os.system(cmd0)
    save_loc = '../logs/fit/fmnist/' + '{}_epochs_'.format(epochs) + \
               '{}_opt_'.format(optimizer_shortform) + \
               '{}_loss_'.format(loss_function_shortform) + \
               '{}_batchsize'.format(batchsize)
    
    # cmd1 = '''python3 tensorflow_exps/build.py --file_info "{'save_loc' : '../'}" 
    # --params "{'normalise_method': 'pixel', 'tf_compile': {'optimizer': 'sgd', 'loss': 'sparse_categorical_crossentropy', 'metrics': ['accuracy']}}"'''
    cmd1 = '''python3 tensorflow_exps/build.py --file_info "{'save_loc' : '''
    cmd2 = '''}" --params "{'normalise_method': 'pixel', 'tf_compile': {'optimizer': '''
    cmd3 = ''', 'loss': '''
    cmd4 = ''', 'metrics': ['accuracy']}}"'''

    cmd = cmd1 +  "'" + save_loc + "'" +  cmd2 + "'" + optimizer + "'"+ cmd3 + "'" +  loss_function +  "'" + cmd4
    
    os.system(cmd)
    os.chdir("./RShiny/")
    return(save_loc)
