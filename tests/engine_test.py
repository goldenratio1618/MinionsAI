from minionsai.engine import BOARD_SIZE, Board, Unit
from minionsai.unit_type import ZOMBIE


def test_board_copy():
    gys = [(2, 3), (1, 1), (1, 0)]
    water = [(1, 2), (2, 2)]
    b = Board(graveyard_locs=gys, water_locs=water)
    # Add some zombies
    b.board[1][1].unit = Unit(0, ZOMBIE)
    b.board[3][3].unit = Unit(0, ZOMBIE)

    b_copy = b.copy()
    # Damage one - this shouldn't carry over to the copy.
    b.board[1][1].unit.curr_health = 1
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            assert b.board[i][j].is_graveyard == b_copy.board[i][j].is_graveyard
            assert b.board[i][j].is_water == b_copy.board[i][j].is_water
            if b.board[i][j].unit is None:
                assert b_copy.board[i][j].unit is None
    assert b_copy.board[1][1].unit.type.name == ZOMBIE.name
    assert b_copy.board[3][3].unit.type.name == ZOMBIE.name

    # Check that the damage didn't carry over.
    assert b_copy.board[1][1].unit.curr_health == ZOMBIE.defense

test_board_copy()