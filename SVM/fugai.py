# coding=utf-8

# 要覆盖的州
states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])

# 广播台清单
stations = {}
stations["kone"] = set(["id", "nv", "ut"])
stations["ktwo"] = set(["wa", "id", "mt"])
stations["kthree"] = set(["or", "nv", "ca"])
stations["kfour"] = set(["nv", "ut"])
stations["kfive"] = set(["ca", "az"])

final_stations = set() # 存储最终选择的广播台

while states_needed:  # 只要没有全部覆盖完
  best_station = None
  states_covered = set()  # 存储已经覆盖的州
  for station, states in stations.items(): # items()存储键值(广播台和相应的覆盖州)
    covered = states_needed & states  # 集合的交集，判断还未覆盖的州与此广播台的交集个数
    if len(covered) > len(states_covered): # 如果当前广播台州交集的个数大于当前要覆盖的州
      best_station = station  # 就替换为最优的广播台
      states_covered = covered  # 替换已经覆盖的州

  states_needed -= states_covered # 从要覆盖的州中减去已经覆盖过的(集合相减)
  print ('states_needed:',states_needed)
  final_stations.add(best_station) # 添加最优的广播台

print (final_stations)