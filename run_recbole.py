# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2020/10/3, 2020/10/1
# @Author : Yupeng Hou, Zihan Lin
# @Email  : houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn


import argparse
import time
from recbole.quick_start import run_recbole
import datetime,time

# if __name__ == '__main__':
#     # # timestamp = 2546272000111
#     timeStamp_checkpoint = 1546272000
#     timeArray = time.localtime(timeStamp_checkpoint)
#     checkpoint = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
#     print("checkpoint：%s" % checkpoint)

    # tss1 = '2019-01-01 00:00:00'
    # # 转为时间数组
    # timeArray = time.strptime(tss1, "%Y-%m-%d %H:%M:%S")
    # print(timeArray)
    # # timeArray可以调用tm_year等
    # print(timeArray.tm_year)
    # print(timeArray.tm_yday)
    # # 转为时间戳
    # timeStamp = int(time.mktime(timeArray))
    # print(timeStamp)

if __name__ == '__main__':

    begin = time.time()
    parameter_dict = {
        # 'neg_sampling': None,
        #'neg_sampling': {'popularity': 1},
        'gpu_id': 0
        # 'attribute_predictor':'not',
        # 'attribute_hidden_size':"[256]",
        # 'fusion_type':'gate',
        # 'seed':212,
        # 'n_layers':4,
        # 'n_heads':1
    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='AMIM', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=' ml-100k.yaml', help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole(model=args.model,
                dataset=args.dataset,
                config_file_list=config_file_list,
                config_dict=parameter_dict)
    end = time.time()
    print(end - begin)

# if __name__ == '__main__':
#     a = [1,2,3,4]
#     print(a[:2])