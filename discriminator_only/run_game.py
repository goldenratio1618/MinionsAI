from engine import Game

def run_game(game_kwargs, agents):
    game = Game(**game_kwargs)
    while not game.done:
        agents[game.active_player_color].act(game)
        game.next_turn()
    return game.winner