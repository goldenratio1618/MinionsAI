import abc
import random

from MinionsAI.action import Action, SpawnAction, MoveAction, FinishPhaseAction, EndTurnAction
from MinionsAI.engine import Game, BOARD_SIZE, Phase

@abc.ABC
class Agent():
    @abc.abstractmethod
    def act(self, game: Game) -> Action:
        raise NotImplementedError()

class CLIAgent(Agent):
    def act(self, game: Game) -> Action:
        # Do stuff with parse_input(yellow)
        raise NotImplementedError()

class RandomAIAgent(Agent):
    def act(self, game: Game) -> Action:
        if game.phase == Phase.MOVE:
            if random.random() < 0.1:
                return FinishPhaseAction()
            xi = random.randrange(0, BOARD_SIZE)
            yi = random.randrange(0, BOARD_SIZE)
            xf = random.randrange(0, BOARD_SIZE)
            yf = random.randrange(0, BOARD_SIZE)
            return MoveAction((xi, yi), (xf, yf))
        elif game.phase == Phase.SPAWN:
            if random.random() < 0.02 or game.money(game.active_player_color) < 2:
                return FinishPhaseAction()
            else:
                x = random.randrange(0, BOARD_SIZE)
                y = random.randrange(0, BOARD_SIZE)
                return SpawnAction(1, (x, y))
        elif game.phase == Phase.FINISH:
            return EndTurnAction()
