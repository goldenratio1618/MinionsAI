from flask import Flask, redirect, render_template, url_for
from models import metadata
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

db_name = 'evals.db'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + db_name

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(metadata=metadata)
db.init_app(app)
from models import Env, Agent, Game

@app.route('/')
def render():
   return render_template('home.html', envs = db.session.query(Env).all() )

# Add a test env:
@app.route('/test')
def add_env():
    # only add it if there aren't any
    if db.session.query(Env).count() == 0:
        env = Env(name='test', game_python_path="MinionsAI.evals.test_game.Game", game_kwargs='{}')
        db.session.add(env)
        db.session.commit()
    return render_template('home.html', envs =  db.session.query(Env).all() )

@app.route('/env/<name>/view')
def env_view(name):
    env = db.session.query(Env).filter_by(name=name).first()
    agents = db.session.query(Agent).filter_by(env_id=env.id).all()
    agents = sorted(agents, key=lambda x: x.trueskill)
    return render_template('env.html', env=env, agents=agents)

@app.route('/env/<name>/upload')
def upload(name):
    # TODO
    env = db.session.query(Env).filter_by(name=name).first()
    previous_agents = len(env.agents)
    agent = Agent(name=f'test_agent_{previous_agents}', env=env, trueskill=0, queued_games=100)
    db.session.add(agent)
    db.session.commit()
    return redirect(url_for('env_view', name=env.name))

@app.route('/agent/<agent>/run/<n>')
def queue_games(agent, n):
    agent = db.session.query(Agent).filter_by(name=agent).first()
    agent.queued_games += int(n)
    db.session.commit()
    return redirect(url_for('env_view', name=agent.env.name))

if __name__ == '__main__':
    app.run(debug=True)