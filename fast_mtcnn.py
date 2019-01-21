from subprocess import PIPE, Popen
import sys
import os

def fast_mtcnn():
  path = os.path.dirname(__file__)
  print('%s/mtcnn_local/mtcnn_local' % path)
  p = Popen(['%s/mtcnn_local/mtcnn_local' % path, '%s/mtcnn_local/models/' % path], stdin=PIPE, stdout=PIPE)
  def detect(img):
    p.stdin.write(img + "\n")
    out = p.stdout.readline().strip()
    if "Initialize OpenCL " in out:
      out = p.stdout.readline().strip()
    try:
        boxes = eval(out)
    except:
        print("cannot detect any face for %s" % img)
        return []
    print("generate %d boxes for img %s" % (len(boxes), img))
    return boxes
  return detect

if __name__ == '__main__':
    mtcnn = fast_mtcnn()
    boxes = mtcnn(sys.argv[1])
    print(boxes)
