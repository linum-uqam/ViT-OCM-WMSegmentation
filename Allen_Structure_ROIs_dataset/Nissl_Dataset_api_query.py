import urllib.request, json 
with urllib.request.urlopen("http://api.brain-map.org/api/v2/data/query.json?criteria=model::SectionDataSet,rma::criteria,[failed$eqfalse],products[id$eq1],treatments[name$eqNISSL],plane_of_section[name$eq%27coronal%27],rma::options[start_row$eq0][num_rows$eq1000]") as url:
    data = json.loads(url.read().decode())
    print(len(data['msg']))
msg = data['msg']
Nissl_Dataset_list = []
for i in range(len(msg)):
  Nissl_Dataset_list.append(msg[i]["id"])
print(Nissl_Dataset_list[0:100:1])
