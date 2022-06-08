from minionsai.action import MoveAction
from minionsai.agent import RandomAIAgent
from minionsai.engine import BOARD_SIZE, Board, Unit, Game, print_n_games
from minionsai.unit_type import ZOMBIE


def test_board_copy():
    gys = [(2, 3), (1, 1), (1, 0)]
    water = [(1, 2), (2, 2)]
    b = Board(graveyard_locs=gys, water_locs=water)
    # Add some zombies
    b.board[1][1].unit = Unit(0, ZOMBIE)
    b.board[3][3].unit = Unit(0, ZOMBIE)
    # Damage one before the copy
    b.board[1][1].unit.curr_health = 1

    b_copy = b.copy()
    # Damage the other one after.
    b.board[3][3].unit.curr_health = 1
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            assert b.board[i][j].is_graveyard == b_copy.board[i][j].is_graveyard
            assert b.board[i][j].is_water == b_copy.board[i][j].is_water
            if b.board[i][j].unit is None:
                assert b_copy.board[i][j].unit is None
    assert b_copy.board[1][1].unit.type.name == ZOMBIE.name
    assert b_copy.board[3][3].unit.type.name == ZOMBIE.name

    # Check that the damage carried.
    assert b_copy.board[1][1].unit.curr_health == 1
    assert b_copy.board[3][3].unit.curr_health == 2

def check_units_equivalent(u1, u2):
    assert u1.type.name == u2.type.name
    assert u1.color == u2.color
    assert u1.curr_health == u2.curr_health
    assert u1.hasMoved == u2.hasMoved
    assert u1.remainingAttack == u2.remainingAttack
    assert u1.isExhausted == u2.isExhausted
    assert u1.is_soulbound == u2.is_soulbound

def test_unit_encode_json_simple():
    unit = Unit(0, ZOMBIE)
    unit_str = unit.encode_json()
    unit_copy = Unit.decode_json(unit_str)
    check_units_equivalent(unit, unit_copy)

def test_unit_encode_json_complex():
    unit = Unit(1, ZOMBIE)
    unit.curr_health = 1
    unit.hasMoved = True
    unit.remainingAttack = 0
    unit.isExhausted = True
    unit.is_soulbound = True
    unit_str = unit.encode_json()
    unit_copy = Unit.decode_json(unit_str)
    check_units_equivalent(unit, unit_copy)

def test_board_encode_json():
    gys = [(2, 3), (1, 1), (1, 0)]
    water = [(1, 2), (2, 2)]
    b = Board(graveyard_locs=gys, water_locs=water)
    # Add some zombies
    b.board[1][1].unit = Unit(0, ZOMBIE)
    b.board[3][3].unit = Unit(1, ZOMBIE)
    # Give them a bunch of state
    b.board[1][1].unit.curr_health = 1
    b.board[1][1].unit.hasMoved = True
    b.board[3][3].unit.remainingAttack = 1

    json_str = b.encode_json()
    b_copy = Board.decode_json(json_str)
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            assert b.board[i][j].is_graveyard == b_copy.board[i][j].is_graveyard
            assert b.board[i][j].is_water == b_copy.board[i][j].is_water
            if b.board[i][j].unit is None:
                assert b_copy.board[i][j].unit is None

    check_units_equivalent(b.board[1][1].unit, b_copy.board[1][1].unit)
    check_units_equivalent(b.board[3][3].unit, b_copy.board[3][3].unit)

def test_game_encode_json():
    game = Game()
    game.board.board[1][1].unit = Unit(0, ZOMBIE)
    game.next_turn()
    game.process_single_action(MoveAction((1, 1), (2, 1)))
    game_str = game.encode_json()
    game_copy = Game.decode_json(game_str)
    assert game.money == game_copy.money
    assert game.income_bonus == game_copy.income_bonus
    assert game.remaining_turns == game_copy.remaining_turns
    assert game.phase == game_copy.phase
    assert game.active_player_color == game_copy.active_player_color
    for unit, (i, j) in game.units_with_locations():
        check_units_equivalent(unit, game_copy.board.board[i][j].unit)

def test_print_n_games():
    games = [Game() for _ in range(10)]
    print_n_games(games)