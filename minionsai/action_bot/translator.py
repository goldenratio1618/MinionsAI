from minionsai.discriminator_only.translator import Translator, ObservationEnum
from minionsai.engine import Phase

class ActionTranslator(Translator):
    def translate_action(self, game: Game, action: Int):
        x = action // BOARD_SIZE ** 3
        y = (action - BOARD_SIZE * x) // BOARD_SIZE ** 2
        a = (action - BOARD_SIZE ** 3 * x - BOARD_SIZE ** 2 * y) // BOARD_SIZE
        b = action - BOARD_SIZE ** 3 * x - BOARD_SIZE ** 2 * y - BOARD_SIZE * a
        return MoveAction((x,y), (a,b))

    MAX_UNIT_HEALTH = 7
    UNIT_TYPES = ObservationEnum([(u, c, m, a, h) for u in [u.name for u in unitList] for c in [True, False] for m in [True, False] for a in [True, False] for h in range(MAX_UNIT_HEALTH)], none_value=True)

    def translate(self, game: Game):
        board_obs = [] # [num_hexes, 3 (location, terrain, unit_type)]
        for (i, j), hex in game.board.hexes():
            terrain = "graveyard" if hex.is_graveyard else "water" if hex.is_water else "none"
            has_moved = False
            is_exhausted = False
            if hex.unit is None:
                unit_type = self.UNIT_TYPES.NULL
            else:
                unit_type = (hex.unit.type.name, hex.unit.color == game.active_player_color, hex.unit.hasMoved, hex.unit.remainingAttack == 0, hex.unit.curr_health)
            
            board_obs.append([
                self.HEXES.encode((i, j)),
                self.TERRAIN_TYPES.encode(terrain),
                self.UNIT_TYPES.encode(unit_type)
            ])

        phase = (game.phase == Phase.MOVE)
        remaining_turns = game.remaining_turns
        all_money = game.money
        scores = game.get_scores
        # Clip the obs to be within bounds
        remaining_turns = min(remaining_turns, self.MAX_REMAINING_TURNS)
        
        money = min(all_money[game.active_player_color], self.MAX_MONEY)
        opp_money = min(all_money[1 - game.active_player_color], self.MAX_MONEY)
        score_diff = max(min(scores[game.active_player_color] - scores[game.inactive_player_color], self.MAX_SCORE_DIFF), -self.MAX_SCORE_DIFF) + self.MAX_SCORE_DIFF
        # TODO: Should the translator be in charge of calling ObservationEnum.encode()?
        return {
            'board': np.array([board_obs]),
            'phase': np.array([[phase]]),
            'remaining_turns': np.array([[remaining_turns]]),  # shape is [batch, num_items]
            'money': np.array([[money]]),
            'opp_money': np.array([[opp_money]]),
            'score_diff': np.array([[score_diff]])
        }
