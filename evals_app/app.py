from flask import Flask, redirect, render_template, url_for
from models import metadata
from flask_sqlalchemy import SQLAlchemy
import click
from flask.cli import with_appcontext
import os

app = Flask(__name__)

db_name = 'evals.db'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/' + db_name
AGENTS_DIR = 'data/agents/'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

db = SQLAlchemy(metadata=metadata)
db.init_app(app)
from models import Env, Agent, Game

@click.command('init-db')
@with_appcontext
def init_db_command():
    """
    Clear the existing data and create new tables.
    """
    db.create_all()

@app.route('/')
def render():
   return render_template('home.html', envs = db.session.query(Env).all() )

def new_env(name):
    env = Env(name=name)
    os.makedir(os.path.join(AGENTS_DIR, name))
    db.session.add(env)
    db.session.commit()
    return env

def new_agent(env, agent_name, data):
    agent = Agent(env=env, name=data['name'], trueskill=data['trueskill'], trueskill_sigma=data['trueskill_sigma'])
    db.session.add(agent)
    db.session.commit()
    return agent


# Sandbox endpoint for testing.
@app.route('/test')
def add_env():
    # only add it if there aren't any
    if db.session.query(Env).count() == 0:
        new_env('test')
    return render_template('home.html', envs =  db.session.query(Env).all() )

@app.route('/env/<name>/view')
def env_view(name):
    env = db.session.query(Env).filter_by(name=name).first()
    agents = db.session.query(Agent).filter_by(env_id=env.id).all()
    agents = sorted(agents, key=lambda x: x.trueskill)
    return render_template('env.html', env=env, agents=agents)

@app.route('/env/<name>/upload')
def upload(env_name):
    # TODO
    env = db.session.query(Env).filter_by(name=env_name).first()
    previous_agents = len(env.agents)
    agent = Agent(name=f'test_agent_{previous_agents}', env=env, trueskill=0, queued_games=1000)
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