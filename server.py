from flask import Flask, request
from json import dumps, loads
from dispatcher import Dispatcher

app = Flask(__name__)

# @app.route('/push_root', methods=['POST'])
# def push_root():
# 	assert 'hash' in request.values.keys()
# 	hash_ = request.values['hash']
# 	txhash = push(hash_)

# 	return dumps({"txhash": txhash})

@app.route('/get_job', methods=['GET'])
def send_job():
	try:
		job_id, params = dispatcher.get_job()
		print(job_id, params)
		return dumps({'status': 1,
					  'job_id': job_id,
					  'params': params})

	except IndexError:
		return dumps({'status': 0})

@app.route('/post_results', methods=['POST'])
def accept_results():
	assert 'output' in request.values.keys()
	output = loads(request.values['output'])
	print(output)
	job_id = output['job_id']
	dispatcher.finish_job(job_id, output)

	return dumps({'status': 1})

if __name__ == "__main__":
	dispatcher = Dispatcher()
	dispatcher.init_state()
	for i in [100, 200, 300, 400]:
		dispatcher.add_job(i)

	app.run(debug=True)