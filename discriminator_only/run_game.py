from engine import Game

def run_game(game_kwargs, agents):
    game = Game(**game_kwargs)
    while not game.done:
        agents[game.active_player_color].act(game)
        # print("Turn taken:")
        # game.pretty_print()
        game.next_turn()
        # print("===================================")
    return game.winner