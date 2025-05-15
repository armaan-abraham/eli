import logging
import boto3
import time
from io import StringIO

class S3LogHandler(logging.Handler):
    def __init__(self, bucket, key_prefix, flush_interval=60):
        super().__init__()
        self.bucket = bucket
        self.key_prefix = key_prefix
        self.buffer = StringIO()
        self.s3_client = boto3.client('s3')
        self.last_flush = time.time()
        self.flush_interval = flush_interval
        
    def emit(self, record):
        msg = self.format(record)
        self.buffer.write(msg + '\n')
        
        # Periodically flush to S3
        if time.time() - self.last_flush > self.flush_interval:
            self.flush()
        
    def flush(self):
        if self.buffer.tell() == 0:
            return
        self.buffer.seek(0)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        log_key = f"{self.key_prefix}/logs/{timestamp}.log"
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=log_key,
                Body=self.buffer.getvalue(),
                ContentType='text/plain'
            )
            self.buffer = StringIO()
            self.last_flush = time.time()
        except Exception as e:
            # Log error but don't stop execution
            print(f"Failed to upload logs to S3: {str(e)}")
        
    def close(self):
        self.flush()
        super().close()

def configure_logging(bucket, key_prefix):
    """
    Configure logging to both console and S3.
    
    Args:
        bucket: S3 bucket name
        key_prefix: Prefix for log files in S3
        
    Returns:
        S3LogHandler instance for later cleanup
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    # Add S3 handler
    s3_handler = S3LogHandler(bucket, key_prefix)
    s3_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(s3_handler)
    
    return s3_handler

def log_progress(current, total):
    """
    Log progress information.
    
    Args:
        current: Current number of samples processed
        total: Total number of samples to process
    """
    percentage = (current / total) * 100 if total > 0 else 0
    logging.getLogger().info(f"Progress: {current}/{total} samples processed ({percentage:.2f}%)") 