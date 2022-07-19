import numpy as np
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
        actions =  [(x, 0) for x in range(1, len(self.list))]
        qs = [q for q, l in self.list[1:]]
        return obs, actions, qs

    def take_action(self, action) -> None:
        """
        Move along the tree to a new location.

        actions are (idx, 0) so that they are tuples of 2 ints like the real data.
        """
        _, self.list = self.list[action[0]]

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
    all_actions, node_pointer, extra_training_data, trajectory = search.run_trajectory()
    assert node_pointer.hash_node() == 'd'
    assert all_actions == [(2, 0), (1, 0)]
    assert extra_training_data is None
    assert trajectory.obs == ['a', 'c']
    assert trajectory.actions == [(2, 0), (1, 0)]
    assert trajectory.maxq[1:] == [2.1]

    # Second trajectory should go a->b
    all_actions, node_pointer, extra_training_data, trajectory = search.run_trajectory()
    assert node_pointer.hash_node() == 'b'
    assert all_actions == [(1, 0),]
    assert extra_training_data is None
    assert trajectory.obs == ['a']
    assert trajectory.actions == [(1, 0)]
    assert trajectory.maxq[1:] == []

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
    NODE_A = {'board': np.array([1.0])}   # Make this obs be a real dict, since it will be inspected by the tree search. Really all the rest should be too. 
    tree = [NODE_A,
            (1.0, ['b']),
            (2.0, ['c', 
                (2.1, subtree),
                (0.5, ['e'])
                ]),
            (1.5, subtree)
            ]
    search = DepthFirstTreeSearch(lambda: MockNode(tree))
    # First trajectory should go a->c->s->u->w
    all_actions, node_pointer, extra_training_data, trajectory = search.run_trajectory()
    assert node_pointer.hash_node() == 'w'
    assert all_actions == [(2, 0), (1, 0), (1, 0), (2, 0)]
    assert extra_training_data is None
    assert trajectory.obs == [NODE_A, 'c', 's', 'u']
    assert trajectory.actions == [(2, 0), (1, 0), (1, 0), (2, 0)]
    assert trajectory.maxq[1:] == [2.1, 1.9, 2.2]

    # Second trajectory should go a->s <restart> a->b
    all_actions, node_pointer, extra_training_data, trajectory = search.run_trajectory()
    assert node_pointer.hash_node() == 'b'
    assert all_actions == [(1, 0),]
    assert trajectory.obs == [NODE_A]
    assert trajectory.actions == [(1, 0)]
    assert trajectory.maxq[1:] == []

    # check that extra_training_data.obs == [NODE_A]
    assert len(extra_training_data.obs.keys()) == 1
    assert extra_training_data.obs["board"] == NODE_A["board"]

    np.testing.assert_equal(extra_training_data.actions, [(3, 0)])
    assert extra_training_data.next_maxq == [1.9,]
