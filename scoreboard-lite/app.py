from flask import Flask, redirect, render_template, request, url_for
from util import list_agents, list_envs, env_dir, format_timedelta, env_agents_dir, env_deleted_agents_dir, env_scores_file
import zipfile
import tempfile
import datetime
import os
import shutil

from minionsai.trueskill_worker import read_scores
from minionsai.agent import RandomAIAgent, NullAgent

from scoreboard_envs import ENVS

UPLOAD_PASSWORD = 'shadowL0rd'

app = Flask(__name__)

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True

def verify_envs_setup():
    print("Verifying envs directories...")
    for env_name in ENVS:
        if not os.path.isdir(env_dir(env_name)):
            # Make it
            print(f"  {env_name} ... creating")
            os.makedirs(env_dir(env_name))
            os.makedirs(env_agents_dir(env_name))
            os.makedirs(env_deleted_agents_dir(env_name))
            # Create new agents
            RandomAIAgent().save(os.path.join(env_agents_dir(env_name), 'random_agent'))
            NullAgent().save(os.path.join(env_agents_dir(env_name), 'null_agent'))

        else:
            print(f"  {env_name} ... exists")

@app.route('/')
def render():
    return render_template('home.html', envs = list_envs() )

@app.route('/env/<env_name>/view')
def env_view(env_name):
    scores, last_update = read_scores(env_scores_file(env_name))
    agent_names = [a['name'] for a in scores]
    agents = [[agent['name'], f"{agent['trueskill']:.1f}", agent['games_played'], agent['ok'], agent['crashes']] for agent in sorted(scores, key = lambda x: x['trueskill'], reverse=True)]
    for agent in list_agents(env_name):
        if agent not in agent_names:
            agents.append([agent, "N/A", 0, "N/A", "N/A"])

    # Calculate last updaet time of the file
    if last_update is not None:
        time_since_last_update = format_timedelta(datetime.datetime.now() - last_update)
        last_update = last_update.strftime("%Y-%m-%d %H:%M:%S")
    else:
        time_since_last_update = "N/A"
        last_update = "N/A"

    return render_template('env.html', env_name=env_name, agents=agents, last_update=last_update, time_since_last_update=time_since_last_update)

def zip_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'zip'

@app.route('/env/<env_name>/upload', methods=['GET', 'POST'])
def upload(env_name, error=''):
    if error == '' and request.method == 'POST':
        if request.values['password'] != UPLOAD_PASSWORD:
            return upload(env_name, error='Wrong password (ask David for password)')
        # check if the post request has the file part
        if 'file' not in request.files:
            return upload(env_name, error='No file part')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return upload(env_name, error='No file selected')
        if not zip_file(file.filename):
            return upload(env_name, error='Please submit a zip file (filename.zip)')
        agent_name = file.filename.rsplit('.', 1)[0]
        print(f"Uploading {agent_name} ({file.filename})")
        if os.path.exists(os.path.join(env_agents_dir(env_name), agent_name)):
            return upload(env_name, error=f"Agent {agent_name} already exists")
        if file:
            temp_location = tempfile.gettempdir()
            file.save(os.path.join(temp_location, file.filename))
            with zipfile.ZipFile(os.path.join(temp_location, file.filename), 'r') as zip_ref:
                final_dest = env_agents_dir(env_name)
                print(f"Received new agent; extracting to {final_dest}")
                zip_ref.extractall(final_dest)
            os.remove(os.path.join(temp_location, file.filename))
            return redirect(url_for('env_view', env_name=env_name))
    return render_template('agent_upload.html', error=error)

@app.route("/env/<env_name>/agent/<agent_name>/crashes")
def agent_crashes(env_name, agent_name):
    all_files = os.listdir(os.path.join(env_agents_dir(env_name), agent_name))
    crashes = [f for f in all_files if 'stacktrace' in f]
    crashes.sort(key = lambda x: x.rsplit('_', 1)[1], reverse=True)
    latest_3_with_contents = [(f, open(os.path.join(env_agents_dir(env_name), agent_name, f)).read()) for f in crashes[:3]]
    return render_template('agent_crashes.html', env_name=env_name, agent_name=agent_name, crashes=latest_3_with_contents, num_crashes=len(crashes))

@app.route("/env/<env_name>/agent/<agent_name>/view")
def agent_view(env_name, agent_name):
    scores, _ = read_scores(env_scores_file(env_name))
    matching_agents = [a for a in scores if a['name'] == agent_name]
    if len(matching_agents) == 0:
        all_agents = list_agents(env_name)
        if agent_name in all_agents:
            return render_template('agent_view.html', env_name=env_name, agent={
                'name': agent_name,
                'trueskill': 0,
                'trueskill_sigma': 0,
                'games_played': 0,
                'ok': "N/A",
                'crashes': 0
            })
        return redirect(url_for('env_view', env_name=env_name))
    agent = matching_agents[0]
    return render_template('agent_view.html', env_name=env_name, agent=agent)

@app.route("/env/<env_name>/agent/<agent_name>/delete")
def agent_delete(env_name, agent_name):
    shutil.move(os.path.join(env_agents_dir(env_name), agent_name), os.path.join(env_deleted_agents_dir(env_name)))
    return env_view(env_name)

if __name__ == '__main__':
    verify_envs_setup()
    app.run(debug=True)