import glob
import csv
import argparse

def findgraphlist(FLAGS):
    graph_dir = FLAGS.dir_input_graphs
    graph_lists = glob.glob(graph_dir)
    return graph_lists

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-I', '--dir_input_graphs', type=str,\
                      default = '/scratch/users/5775project/PlacementML4/graph/dataset/data/llvm/txt/*',\
                      help='input llvm graphs,\
                      String. Default:/scratch/users/5775project/PlacementML4/graph/dataset/data/llvm/txt/*')
  FLAGS, unknown=parser.parse_known_args()



  graph_lists = findgraphlist(FLAGS)
  index = 0
  mul_index = []
  node_vec_map = dict()
  template = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
  for i in range(14):
    template[i] = 1
    node_vec_map[i] = template[:]
    template[i] = 0
  node_feat = []
  adj_list = []
  mul_index = []
  mul_over = []
  mul_under = []
  mul_normal = []
  for g in graph_lists:
    with open(g) as f:
      node_index = dict()
      for i, line in enumerate(f):
        info = line.strip().split()
        if i == 0:
          continue
        elif i == 1:
          node_class = info[0]
        elif i == 2:
          num_nodes = int(info[0])
        elif i > 2 and i < num_nodes + 3:
          index += 1
          node_index[int(info[0])] = index
          node_vec = node_vec_map[int(info[1])]
          if int(info[1]) == 5:
            if node_class == 'OverEstimate':
              mul_over.append(index)
            elif node_class == 'UnderEstimate':
              mul_under.append(index)
            else:
              mul_normal.append(index)
          node_feat.append([index] + node_vec + [node_class])
        elif i > num_nodes + 2:
          adj_list.append([node_index[int(info[0])], node_index[int(info[1])]])
  mul_index += [mul_over] + [mul_under] + [mul_normal] 
  print len(mul_over), len(mul_under), len(mul_normal)

  with open("hls/node_feats.csv", "w") as fp:
      writer = csv.writer(fp, delimiter=' ')
      writer.writerows(node_feat)

  with open("hls/connect.csv", "w") as ff:
      writer = csv.writer(ff, delimiter=' ')
      writer.writerows(adj_list)

  with open("hls/mulIndex.csv", "w") as mm:
      writer = csv.writer(mm, delimiter=' ')
      writer.writerows(mul_index)
