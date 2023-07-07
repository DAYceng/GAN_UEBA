from elasticsearch import Elasticsearch

from gan4netflow.implementations.datapreprocessing import Preprocessing

es = Elasticsearch("http://172.23.6.213:9200")
test_Preprocessing = Preprocessing()

hits_list = Preprocessing.get_log_from_es(esclient=es)
print(hits_list)
# print(json.loads(hits_list[0]))
print(len(hits_list))
