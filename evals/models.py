from sqlalchemy import MetaData, Column, Integer, String, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
metadata = MetaData()
Base = declarative_base(metadata=metadata)


class Env(Base):
    __tablename__ = 'envs'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    game_python_path = Column(String, default='MinionsAI.engine.Game')
    game_kwargs = Column(String)
    agents = relationship('Agent', backref='env', lazy=True)
    games = relationship('Game', backref='env', lazy=True)

    def build(self):
        # hack hack
        if self.game_python_path == "MinionsAI.engine.Game":
            from MinionsAI.engine import Game
            cls = Game
        elif self.game_python_path == "MinionsAI.evals.test_game.Game":
            from MinionsAI.evals.test_game import Game
            cls = Game
        else:
            raise ValueError("I guess the lifetime of this hack has expired.")
    
        kwargs = eval(self.game_kwargs)
        return cls(**kwargs)


class Agent(Base):
    __tablename__ = 'agents'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    trueskill = Column(Float)
    trueskill_sigma = Column(Float)
    queued_games = Column(Integer)
    env_id = Column(Integer, ForeignKey('envs.id'))
    games_p1_id = Column(Integer, ForeignKey('games.id'))
    games_p1 = relationship('Game', backref='agent_p1', lazy=True, foreign_keys=[games_p1_id])

    games_p2_id = Column(Integer, ForeignKey('games.id'))
    games_p2 = relationship('Game', backref='agent_p2', lazy=True, foreign_keys=[games_p2_id])

    def build(self):
        # TODO
        return None
    
class Game(Base):
    __tablename__ = 'games'
    id = Column(Integer, primary_key=True)
    env_id = Column(Integer, ForeignKey('envs.id'))
    num_wins_p1 = Column(Integer)
    p1_id = Column(Integer, ForeignKey('agents.id'))
    p2_id = Column(Integer, ForeignKey('agents.id'))