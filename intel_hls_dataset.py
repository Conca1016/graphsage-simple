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
                      default = '/scratch/users/cd574/intel_hls/quartus_17.0/hls/G2HLS/G2HLS/data/randgraph/*',\
                      help='input llvm graphs,\
                      String. Default:/scratch/users/5775project/PlacementML4/graph/dataset/data/llvm/txt/*')
  FLAGS, unknown=parser.parse_known_args()



  graph_lists = findgraphlist(FLAGS)
  index = -1
  mul_index = []
  node_vec_map = dict()
  template = [0, 0, 0, 0, 0]
  for i in range(4):
    template[i] = 1
    node_vec_map[i] = template[:]
    template[i] = 0
  node_feat = []
  adj_list = []
  mul_index = []
  mul_before = []
  mul_after = []
  for g in graph_lists:
    with open(g) as f:
      for i, line in enumerate(f):
        info = line.strip().split()
        if i > 1 and int(info[1]) == 3:
          if i < 9:
            node_class = 'MULT_Before_AddChain'
          else:
            node_class = 'MULT_After_AddChain'
    with open(g) as f:
      flag = 0
      node_index = dict()
      for i, line in enumerate(f):
        info = line.strip().split()
        if i == 0:
          continue
        elif i == 1:
          num_nodes = int(info[0])
        elif i > 1 and i < num_nodes + 2:
          index += 1
          node_index[int(info[0])] = index
          if int(info[1]) == 0:
            node_vec = [0, 0, 0, 1]
          elif int(info[1]) == 3:
            node_vec = [0, 1, 0, 0]
            flag = 1
          elif flag == 0:
            node_vec = [1, 0, 0, 0]
          elif flag == 1:
            node_vec = [0, 0, 1, 0]
          if int(info[1]) == 3:
            if node_class == 'MULT_Before_AddChain':
              mul_before.append(index)
            elif node_class == 'MULT_After_AddChain':
              mul_after.append(index)
          node_feat.append([index] + node_vec + [node_class])
        elif i > num_nodes + 1:
          adj_list.append([node_index[int(info[0])], node_index[int(info[1])]])
  mul_index += [mul_before] + [mul_after] 
  print len(mul_before), len(mul_after)

  with open("intel_hls/node_feats.csv", "w") as fp:
      writer = csv.writer(fp, delimiter=' ')
      writer.writerows(node_feat)

  with open("intel_hls/connect.csv", "w") as ff:
      writer = csv.writer(ff, delimiter=' ')
      writer.writerows(adj_list)

  with open("intel_hls/mulIndex.csv", "w") as mm:
      writer = csv.writer(mm, delimiter=' ')
      writer.writerows(mul_index)
