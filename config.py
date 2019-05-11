
n_heavy_hitters = 20 # make this smaller

time_between_train = 100 # so that you may actually have that many

f_list = [1, 0.99, 0.9, 0.75, 0.5,  0.3, 0.25, 0.1]

decay = f_list[1]

n_gradient_updates = 4

len_stream = 1000

half_batch=16

n_layers=2

limit_light = True