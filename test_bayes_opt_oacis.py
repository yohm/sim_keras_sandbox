import sys,math
import numpy as np
from bayes_opt import BayesianOptimization
import oacis

sim = oacis.Simulator.find_by_name("keras_mnist_mlp")
base_param = {
        #"dense1_size": 128, "dense2_size": 128,
        "dropout1_prob": 0.2, "dropout2_prob": 0.2,
        "batch_size": 1024, "epochs": 10
        }
host = oacis.Host.find_by_name("localhost")
host_param = {}
num_runs = 5
# host_group = oacis.HostGroup.find_by_name("all_desrt")

def _param_to_ps(p1,p2):
    dense1_size = int(2**p1)
    dense2_size = int(2**p2)
    y = {"dense1_size": dense1_size, "dense2_size": dense2_size}
    ps = sim.find_or_create_parameter_set( {**base_param, **y} )
    return ps

def f(p1,p2):
    ps = _param_to_ps(p1,p2)
    runs = ps.find_or_create_runs_upto( 3, submitted_to=host )
    oacis.OacisWatcher.await_ps(ps)
    loss = ps.average_result("test_loss")[0]
    print("loss %f @ %s" % (loss, (p1,p2) ) )
    return -loss

def search_optimum():
    bo = BayesianOptimization(f, {'p1': (4, 9), 'p2': (4, 9)}, random_state=1234)
    def _set_init_points():
        query = { "v."+k:v for k,v in base_param.items() }
        pss = sim.parameter_sets().where(query)
        p1_list = []
        p2_list = []
        #target_list = []
        for ps in pss:
            p1_list.append( math.log2(ps.v()["dense1_size"]) )
            p2_list.append( math.log2(ps.v()["dense2_size"]) )
            #target_list.append( ps.average_result("test_loss")[0] )
        print(p1_list, p2_list)#, target_list)
        #bo.initialize( {'target': target_list, "p1": p1_list, "p2": p2_list} )
        bo.explore( {"p1": p1_list, "p2": p2_list} )
    _set_init_points()
    #bo.maximize(n_iter=1, kappa=2)
    bo.maximize(init_points=0, n_iter=10, kappa=2)
    opt_dense1_size = int(2**bo.res['max']['max_params']['p1'])
    opt_dense2_size = int(2**bo.res['max']['max_params']['p2'])
    print(bo.res['max'], opt_dense1_size, opt_dense2_size)
    print(bo.res['all'])

w = oacis.OacisWatcher()
w.async( search_optimum )
w.loop()

