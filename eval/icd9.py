import csv
import json
from collections import *

class Node(object):
  def __init__(self, depth, code, descr=None):
    self.depth = depth
    self.descr = descr or code
    self.code = code
    self.parent = None
    self.children = []

  def add_child(self, child):
    if child not in self.children:
      self.children.append(child)

  def search(self, code):
    if code == self.code: return [self]
    ret = []
    for child in self.children:
      ret.extend(child.search(code))
    return ret

  def find(self, code):
    nodes = self.search(code)
    if nodes:
      return nodes[0]
    return None

  @property
  def root(self):
    return self.parents[0]

  @property
  def description(self):
    return self.descr

  @property
  def codes(self):
    return map(lambda n: n.code, self.leaves)

  @property
  def parents(self):
    n = self
    ret = []
    while n:
      ret.append(n)
      n = n.parent
    ret.reverse()
    return ret


  @property
  def leaves(self):
    leaves = set()
    if not self.children:
      return [self]
    for child in self.children:
      leaves.update(child.leaves)
    return list(leaves)

  # return all leaf notes with a depth of @depth
  def leaves_at_depth(self, depth):
    return filter(lambda n: n.depth == depth, self.leaves)

  @property
  def siblings(self):
    parent = self.parent
    if not parent:
      return []
    return list(parent.children)

  def __str__(self):
    return '%s\t%s' % (self.depth, self.code)

  def __hash__(self):
    return hash(str(self))


class ICD9(Node):
  def __init__(self, codesfname):
    # dictionary of depth -> dictionary of code->node
    self.depth2nodes = defaultdict(dict)
    super(ICD9, self).__init__(-1, 'ROOT')

    with file(codesfname, 'r') as f:
      allcodes = json.loads(f.read())
      self.process(allcodes)

  def process(self, allcodes):
    for hierarchy in allcodes:
      self.add(hierarchy)

  def get_node(self, depth, code, descr):
    d = self.depth2nodes[depth]
    if code not in d:
      d[code] = Node(depth, code, descr)
    return d[code]

  def add(self, hierarchy):
    prev_node = self
    for depth, link in enumerate(hierarchy):
      if not link['code']: continue

      code = link['code']
      descr = 'descr' in link and link['descr'] or code
      node = self.get_node(depth, code, descr)
      node.parent = prev_node
      prev_node.add_child(node)
      prev_node = node


if __name__ == '__main__':
  tree = ICD9('codes.json')
  counter = Counter(map(str, tree.leaves))
  import pdb
  pdb.set_trace()
