from vector_service_pb2 import TextRequest, VectorResponse
import vector_service_pb2_grpc
import grpc
import concurrent.futures
import tensorflow as tf
import tensorflow_text
import numpy as np
import os


model_path = "universal-sentence-encoder-multilingual-large_3"
model = tf.saved_model.load(model_path)

port = os.environ.get('PORT', '50051')

class VectorService(
    vector_service_pb2_grpc.VectorServiceServicer
):

    def SendVector(self, request_iterator, context):
        for textRequest in request_iterator:
            messages = [textRequest.text]
            embeddings = self.get_message_embdings(messages)
            vectorResult = VectorResponse(
                vector = embeddings
            )
            yield vectorResult
    def get_message_embdings(self, input):
        return np.array(model(input))[0]    




def serve():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    vector_service_pb2_grpc.add_VectorServiceServicer_to_server(
        VectorService(), server
    )
    server.add_insecure_port("[::]:"+port)
    server.start()
    print("all ok")
    server.wait_for_termination()



if __name__ == "__main__":
    serve()




