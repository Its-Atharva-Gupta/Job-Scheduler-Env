import time, random

class Job:
    def __init__(self, current_time):
        self.id = int(time.time()) + current_time + random.randint(3000,9999)
        self.duration = random.randint(1, 10)  # How long the task takes to get completed
        self.deadline = random.randint(5, 25) # How urgent the task is (must be finished befor current_time = deadline)
        self.arrival = current_time + random.randint(1, 33) # When the job will arrive to the agent
        self.is_happening:bool = False


jobs = []
for i in range(3):
    new_job = Job(current_time=i)
    jobs.append(new_job)

print(jobs)

def job_as_json(job: Job):
    return {
        'id' : job.id,
        'duration': job.duration,
        'deadline': job.deadline,
        'arrival': job.arrival,
        'is_happening': job.is_happening
    }

print(job_as_json(jobs[1]))