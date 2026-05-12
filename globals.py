number_SH_properties = [0, 3, 8, 15]
SH_properties_single = None
SH_properties = None

# Start and end indices for the properties tensor from CPU Memory
xyz1 = 0
xyz2 = 3
scales1 = 3
scales2 = 6
rotation1 = 6
rotation2 = 10
features1 = 10
features2 = 13
opacity1 = 13
opacity2 = 14
features_rest1 = 14
features_rest2 = None # This is set after hierarchy is loaded
number_properties = None
range1 = [xyz1, scales1, rotation1, features1, opacity1, features_rest1]
range2 = [xyz2, scales2, rotation2, features2, opacity2, features_rest2]

hierarchy_node_depth = 0
hierarchy_node_parent = 1
hierarchy_node_child_count = 2
hierarchy_node_first_child = 3
hierarchy_node_next_sibling = 4
hierarchy_node_max_side_length = 5