from minionsai.gen_disc.tree_search import NodePointer, DepthFirstTreeSearch

class MockNode(NodePointer):
    """
    Hacky mock tree for testing. Nested lists; each list is a node. 
    First entry of each list is the name of the node.
    Subsequent entires are (q, node)
    e.g.
        ['a', 
            (1.0, ['b']),
            (2.0, ['c', 
                (0.5, ['d']),
                (2.1, ['e'])
            ])
        ]
    
    obs is equal to the name
    actions are just indices.
    """
    def __init__(self, nested_lists_tree):
        self.list = nested_lists_tree

    def hash_node(self):
        """
        Hash of the current node.
        """
        return self.list[0]

    def evaluate_node(self):
        """
        Return the obs of this node, the available actions, and the Q-values of the available actions.
        """
        obs = self.list[0]
        actions =  list(range(1, len(self.list)))
        qs = [q for q, l in self.list[1:]]
        return obs, actions, qs

    def take_action(self, action) -> None:
        """
        Move along the tree to a new location.
        """
        _, self.list = self.list[action]

def test_tree_search_trivial():
    tree = ['a', 
            (1.0, ['b']),
            (2.0, ['c', 
                (2.1, ['d']),
                (0.5, ['e'])
                ])
            ]
    search = DepthFirstTreeSearch(lambda: MockNode(tree))
    # First trajectory should go a->c->d
    all_actions, node_pointer, training_data = search.run_trajectory()
    assert node_pointer.hash_node() == 'd'
    assert all_actions == [2, 1]
    assert training_data['obs'] == ['a', 'c']
    assert training_data['actions'] == [2, 1]
    assert training_data['next_maxq'] == [2.1]

    # Second trajectory should go a->b
    all_actions, node_pointer, training_data = search.run_trajectory()
    assert node_pointer.hash_node() == 'b'
    assert all_actions == [1,]
    assert training_data['obs'] == ['a']
    assert training_data['actions'] == [1]
    assert training_data['next_maxq'] == []

def test_tree_search_alternate_paths():
    subtree = ['s',
                (1.9, ['u',
                    (0.7, ['v',
                        (0.6, ['x']),
                        (2.5, ['y']),
                    ]),
                    (2.2, ['w'])
                ])

    ]
    tree = ['a', 
            (1.0, ['b']),
            (2.0, ['c', 
                (2.1, subtree),
                (0.5, ['e'])
                ]),
            (1.5, subtree)
            ]
    search = DepthFirstTreeSearch(lambda: MockNode(tree))
    # First trajectory should go a->c->s->u->w
    all_actions, node_pointer, training_data = search.run_trajectory()
    assert node_pointer.hash_node() == 'w'
    assert all_actions == [2, 1, 1, 2]
    assert training_data['obs'] == ['a', 'c', 's', 'u']
    assert training_data['actions'] == [2, 1, 1, 2]
    assert training_data['next_maxq'] == [2.1, 1.9, 2.2]

    # Second trajectory should go a->s <restart> a->b
    all_actions, node_pointer, training_data = search.run_trajectory()
    assert node_pointer.hash_node() == 'b'
    assert all_actions == [1,]
    assert training_data['obs'] == ['a', 'a']
    assert training_data['actions'] == [3, 1]
    assert training_data['next_maxq'] == [1.9,]
