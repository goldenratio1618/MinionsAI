from flask import Flask, redirect, render_template, request, url_for
import os
from scoreboard_envs import ENVS
from util import list_agents, list_envs, env_dir, format_timedelta
import zipfile
import tempfile
import csv
from minionsai.agent import RandomAIAgent, NullAgent
import datetime

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

def verify_envs_setup():
    print("Verifying envs directories...")
    for env_name in ENVS:
        if not os.path.isdir(env_dir(env_name)):
            # Make it
            print(f"  {env_name} ... creating")
            os.makedirs(env_dir(env_name))
            # Create new agents
            RandomAIAgent().save(os.path.join(env_dir(env_name), 'random_agent'))
            NullAgent().save(os.path.join(env_dir(env_name), 'null_agent'))

        else:
            print(f"  {env_name} ... exists")

def read_scores(env):
    # scores are stored in scores.csv
    # With columns 'name', 'trueskill', 'trueskill_sigma', 'games_played'
    scores_file = os.path.join(env_dir(env), 'scores.csv')
    # If the file doesn't exist, return an empty list
    if not os.path.isfile(scores_file):
        print(f"No scores file found at {scores_file}")
        return []
    with open(scores_file, 'r') as f:
        reader = csv.reader(f)
        # first line is header; check that it matches our expectations
        header = next(reader)
        if header != ['name', 'trueskill', 'trueskill_sigma', 'games_played']:
            raise Exception(f"Unexpected header in {scores_file}: {header}")
        scores = list(reader)
    # reformat to {'name': name, 'trueskill': trueskill, 'trueskill_sigma': trueskill_sigma, 'games_played': games_played}
    scores = [{'name': s[0], 'trueskill': float(s[1]), 'trueskill_sigma': float(s[2]), 'games_played': int(s[3])} for s in scores]
    return scores

@app.route('/')
def render():
    return render_template('home.html', envs = list_envs() )

@app.route('/env/<env_name>/view')
def env_view(env_name):
    agents = os.listdir(env_dir(env_name))
    agents = [a for a in agents if os.path.isdir(os.path.join(env_dir(env_name), a))]
    scores = read_scores(env_name)
    agent_names = [a['name'] for a in scores]
    agents = [[agent['name'], f"{agent['trueskill']:.1f}", agent['games_played']] for agent in sorted(scores, key = lambda x: x['trueskill'], reverse=True)]
    for agent in list_agents(env_name):
        if agent not in agent_names:
            agents.append([agent, "N/A", 0])

    # Calculate last updaet time of the file
    last_update = datetime.datetime.fromtimestamp(os.path.getmtime(os.path.join(env_dir(env_name), 'scores.csv')))
    time_since_last_update = format_timedelta(datetime.datetime.now() - last_update)

    return render_template('env.html', env_name=env_name, agents=agents, last_update=last_update, time_since_last_update=time_since_last_update)

def zip_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'

@app.route('/env/<env_name>/upload', methods=['GET', 'POST'])
def upload(env_name):
    # TODO - better error message displaying.
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return upload(env_name, error='No file part')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return redirect(request.url)
        if not zip_file(file.filename):
            return redirect(request.url)
        agent_name = file.filename.rsplit('.', 1)[0]
        print(f"Uploading {agent_name} ({file.filename})")
        if os.path.exists(os.path.join(env_dir(env_name), agent_name)):
            print(f"Agent {agent_name} already exists")
            return redirect(request.url)
        if file:
            temp_location = tempfile.gettempdir()
            file.save(os.path.join(temp_location, file.filename))
            with zipfile.ZipFile(os.path.join(temp_location, file.filename), 'r') as zip_ref:
                final_dest = env_dir(env_name)
                print(f"Received new agent; extracting to {final_dest}")
                zip_ref.extractall(final_dest)
            os.remove(os.path.join(temp_location, file.filename))
            return redirect(url_for('env_view', env_name=env_name))
    return render_template('agent_upload.html')

if __name__ == '__main__':
    verify_envs_setup()
    app.run(debug=True)