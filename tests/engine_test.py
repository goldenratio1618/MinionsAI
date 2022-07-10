from calendar import c
from minionsai.action import ActionList, MoveAction, SpawnAction
from minionsai.agent import RandomAIAgent
from minionsai.engine import BOARD_SIZE, Board, Unit, Game, print_n_games
from minionsai.game_util import seed_everything
from minionsai.unit_type import NECROMANCER, ZOMBIE


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

def test_game_unit_states():
    seed_everything(0)
    for _ in range(5):
        game = Game()
        agent = RandomAIAgent()
        while True:
            game.next_turn()
            if game.done:
                break
            # All units should be full health
            for unit, _ in game.units_with_locations(color=game.inactive_player_color):
                assert unit.curr_health == unit.type.defense
            # All active units should have their move and attack.
            for unit, _ in game.units_with_locations(color=game.active_player_color):
                assert unit.hasMoved == False
                assert unit.remainingAttack == unit.type.attack
            # All inactive units should not.
            for unit, _ in game.units_with_locations(color=game.inactive_player_color):
                assert unit.hasMoved == True
                assert unit.remainingAttack == 0
            
            action = agent.act(game.copy())
            game.full_turn(action)

            # No units should have move or attack until we start the next turn.
            for unit, _ in game.units_with_locations():
                assert unit.hasMoved == True
                assert unit.remainingAttack == 0



def test_print_n_games():
    games = [Game() for _ in range(10)]
    print_n_games(games)

def test_game_hash():
    seed_everything(0)
    game = Game()
    game.board.board[1][1].unit = Unit(0, ZOMBIE)
    game.next_turn()
    game.process_single_action(MoveAction((1, 1), (2, 1)))
    game_hash = hash(game)
    second_game_hash = hash(game)
    assert game_hash == second_game_hash
    game_copy_hash = hash(game.copy())
    assert game_hash == game_copy_hash

    seed_everything(0)
    different_game = Game()
    different_game.board.board[1][1].unit = Unit(0, ZOMBIE)
    different_game.next_turn()
    different_game.process_single_action(MoveAction((1, 1), (1, 2)))
    different_game_hash = hash(different_game)
    assert different_game_hash != game_hash

def test_game_same_spawn_vs_move():
    # Test that in the spawn phase, it doesn't matter if a unit moved or not or was spawned.
    seed_everything(0)
    base_game = Game()
    base_game.board.board[0][0].unit = Unit(0, NECROMANCER)
    base_game.board.board[1][0].unit = Unit(0, ZOMBIE)
    base_game.next_turn()

    # Move the zombie and spawn a new one where it was
    moved_game = base_game.copy()
    moved_game.full_turn(ActionList([MoveAction((1, 0), (0, 1))], [SpawnAction(ZOMBIE, (1, 0))]))
    moved_game_hash = hash(moved_game)

    # Spawn a new zombie in the new spot
    no_moved_game = base_game.copy()
    no_moved_game.full_turn(ActionList([], [SpawnAction(ZOMBIE, (0, 1))]))
    no_moved_game_hash = hash(no_moved_game)

    assert moved_game_hash == no_moved_game_hash

    # Should also be the same if the zombie was just there to start with
    seed_everything(0)
    started_there_game = Game()
    started_there_game.board.board[0][0].unit = Unit(0, NECROMANCER)
    started_there_game.board.board[1][0].unit = Unit(0, ZOMBIE)
    started_there_game.board.board[0][1].unit = Unit(0, ZOMBIE)
    started_there_game.money[0] -= 2
    started_there_game.next_turn()
    started_there_game.full_turn(ActionList([], []))
    started_there_game_hash = hash(started_there_game)
    print_n_games([moved_game, started_there_game])
    assert started_there_game_hash == moved_game_hash