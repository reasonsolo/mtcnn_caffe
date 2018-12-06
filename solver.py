import sys
import caffe
import caffe_tools
from caffe_tools.solvers import PlotLossCallback

def count_errors():
    pass

class MtcnnLossCallback(PlotLossCallback):
    def __init__(self, report_interval = 100, save = '', show = False, loss_name = 'loss'):
        super(MtcnnLossCallback, self).__init__(report_interval, save, show)
        self.loss_name = loss_name

    def _get_loss(self, iteration, solver):
        if len(self._iterations) == 0 or self._iterations[-1] != iteration:
            self._iterations.append(iteration)
            self._losses.append(solver.net.blobs[self.loss_name].data)

if __name__ == '__main__':
    solver_prototxt = sys.argv[1]
    max_steps = int(sys.argv[2])

    solver = caffe.SGDSolver(solver_prototxt)
    if len(sys.argv) > 3:
        solver_state = sys.argv[3]
        solver.restore(solver_state)
    for i in range(0, max_steps):
        solver.step(1)
    # callbacks = []
    #report_label_loss = MtcnnLossCallback(1000, 'label_loss.png', True, 'label_loss')
    #report_bbox_loss = MtcnnLossCallback(1000, 'bbox_loss.png', True, 'bbox_loss')
    #callbacks.append({
    #    'callback': PlotLossCallback.report_loss,
    #    'object': report_label_loss,
    #    'interval': 1,
    #})
    #callbacks.append({
    #    'callback': PlotLossCallback.report_loss,
    #    'object': report_bbox_loss,
    #    'interval': 1,
    #})

    # monitoring_solver = caffe_tools.solvers.MonitoringSolver(solver)
    # monitoring_solver.register_callback(callbacks)
    # monitoring_solver.solve(int(sys.argv[2]))
