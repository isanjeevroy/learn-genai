from redis import Redis
from rq import Queue, SimpleWorker

if __name__ == '__main__':
    # Connect to your local Redis server
    connection = Redis(host='localhost', port=6379)
    
    # Define the queues you want to listen to
    queues = [Queue('default', connection=connection)]
    
    # Use SimpleWorker instead of the standard Worker
    worker = SimpleWorker(queues, connection=connection)
    
    print("Starting Windows-compatible RQ SimpleWorker...")
    worker.work()