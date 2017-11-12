import sys
import numpy as np
from bayes_opt import BayesianOptimization
import oacis

sim = oacis.Simulator.find_by_name("keras_mnist_mlp")
base_param = {
        "dense1_size": 128, "dense2_size": 128,
        "dropout1_prob": 0.2, "dropout2_prob": 0.2,
        "batch_size": 1024, "epochs": 10
        }
host = oacis.Host.find_by_name("localhost")
host_param = {}
# host_group = oacis.HostGroup.find_by_name("all_desrt")


def f(p1,p2):
    dense1_size = int(2**p1)
    dense2_size = int(2**p2)
    y = {"dense1_size": dense1_size, "dense2_size": dense2_size}
    ps = sim.find_or_create_parameter_set( {**base_param, **y} )
    print("ps : %s" % ps.v())
    runs = ps.find_or_create_runs_upto( 1, submitted_to=host )
    oacis.OacisWatcher.await_ps(ps)
    loss = ps.average_result("test_loss")[0]
    print("loss %f" % loss)
    return -loss

def search_optimum():
    bo = BayesianOptimization(f, {'p1': (4, 9), 'p2': (4, 9)}, random_state=1234)
    bo.maximize(init_points=5, n_iter=1, kappa=2)
    print(bo.res['max'])
    print(bo.res['all'])

w = oacis.OacisWatcher()
w.async( search_optimum )
w.loop()

