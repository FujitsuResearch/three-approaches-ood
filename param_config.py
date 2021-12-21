import argparse
import pathlib
import json

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default="mnist", type=str, choices=['mnist', 'ilab', 'carcgs', 'daiso'], help="name of dataset ['mnist', 'ilab', 'carcgs', 'daiso']")
    parser.add_argument("-t", "--trial", default=1, type=int, help="number of trials")
    parser.add_argument("-s", "--sampling", default="random", type=str, help="name of sampling method random")
    parser.add_argument("-rw", "--r_weight", default=0.0, type=float, help="the number of regularization weight (lambda)")
    parser.add_argument("-e", "--epoch", default=100, type=int, help="the number of epoch size")
    parser.add_argument("-pi", "--pairing_interval", default=40, type=int, help="the number of pairing (neural sampling) interval")
    parser.add_argument("-name", "--job_name", default="trial", type=str, help="job name")
    parser.add_argument("-id", "--job_id", default=0, type=int, help="the number of job id")
    parser.add_argument("-sd", "--seed", default=0, type=int, help="the number of random seed")
    parser.add_argument("-r", "--rate", default=2, type=int, help="the number of seen rate")
    parser.add_argument("-n", "--norm_type", default=2, type=int, help="the number of p of L_p norm")
    parser.add_argument("-tv", "--run_mode", default="validate", type=str, help="run mode (tune validate)=hyper-parameter tuning or validate")
    parser.add_argument("-bnm", "--batch_momentum", default=0.99, type=float, help="the number of batch normalization momentum")
    parser.add_argument("-ds", "--d_size", help="the numbers of train, validation and test")
    parser.add_argument("-ns", "--num_classes", help="the numbers of classes and nuisance attribute")
    parser.add_argument("-rb", "--r_batchsize", type=int, help="the number of replica batchsize")
    parser.add_argument("-gb", "--g_batchsize", type=int, help="the number of global batchsize")
    parser.add_argument("-lr", "--lr", type=float, help="the number learning rate")
    param = parser.parse_args()
    return param

def init_dict():
    d_key = ['mnist', 'ilab', 'carcgs', 'daiso']
    key = ['d_size','num_classes','r_batchsize','g_batchsize','lr']
    m_value = [[54000, 8000, 8000], [9,9], 64, 256, 0.001]
    i_value = [[18000, 8000, 8000], [6,6], 64, 256, 0.001]
    c_value = [[3400, 450, 800], [10,10], 8, 32, 0.0001]
    d_value = [[800, 200, 400], [5,5], 8, 32, 0.0001]
    mnist = dict(zip(key, m_value))
    ilab = dict(zip(key, i_value))
    carcgs = dict(zip(key, c_value))
    daiso = dict(zip(key, d_value))
    all_dict = dict(zip(d_key, [mnist, ilab, carcgs, daiso]))
    return all_dict

def init_param():
    param = init_parser()
    all_dict = init_dict()
    d_dict = all_dict[param.data]
    if param.d_size == None: param.d_size = d_dict['d_size']
    if param.num_classes == None: param.num_classes = d_dict['num_classes'] 
    if param.r_batchsize == None: param.r_batchsize = d_dict['r_batchsize']
    if param.g_batchsize == None: param.g_batchsize = d_dict['g_batchsize']
    if param.lr == None: param.lr = d_dict['lr']

    # for git commit log 
    """
    git_dir = pathlib.Path('./.git')
    with (git_dir/'HEAD').open('r') as head:
        ref = head.readline().split(' ')[-1].strip()
    with (git_dir/ref).open('r') as git_commit:
        commit_hash = git_commit.readline().strip()
    param.ref = ref
    param.hash = commit_hash
    """
    
    return param

def get_dict_param(param):
    param_dict = param.__dict__
    return param_dict

def main():
    param = init_param()
    param_dict = get_dict_param(param)
    param_json = json.dumps(param_dict, indent=2)
    print(param_json)


if __name__ == "__main__":
    main()
