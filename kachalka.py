import requests
from json import dumps, loads
from time import sleep

base_url = "http://0.0.0.0:5000/"


while True:
	r = requests.get(base_url + "get_job")
	response = loads(r.content)
	if response['status']:
		print("damn son")
		job_id = response['job_id']
		output = {"job_id": job_id,
				  "ayy": "lmao"}

		requests.post(base_url + "post_results", data={'output': dumps(output)})

	else:
		sleep(5)
