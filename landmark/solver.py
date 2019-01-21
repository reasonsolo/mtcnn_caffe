import sys
import caffe

if __name__ == '__main__':
    solver_prototxt = sys.argv[1]
    max_steps = int(sys.argv[2])

    solver = caffe.SGDSolver(solver_prototxt)
    if len(sys.argv) > 3:
        solver_state = sys.argv[3]
        solver.restore(solver_state)
    for i in range(0, max_steps):
        solver.step(1)
