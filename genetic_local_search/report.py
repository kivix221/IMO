import time 
from utils import *
from genetic_local_search.genetic_local_search import genetic_local_search


kroa200_instance = load_instance('./data/kroa200.tsp')
krob200_instance = load_instance('./data/krob200.tsp')
kroa200_distance_matrix = calc_distance_matrix(kroa200_instance)
krob200_distance_matrix = calc_distance_matrix(krob200_instance)


results_kroa200, results_time_kroa200 = [],[]
results_krob200, results_time_krob200 = [],[]


ITER = 10
for _ in range(ITER):
   ########### kroa 200
   duration = time.time()
   x = genetic_local_search(kroa200_distance_matrix,488) #
   duration = time.time() - duration
   
   results_kroa200.append(x)
   results_time_kroa200.append(duration)
   
   ########### krob200
   duration = time.time()
   x = genetic_local_search(krob200_distance_matrix,515) #
   duration = time.time() - duration
   
   
   results_krob200.append(x)
   results_time_krob200.append(duration)
   
min_result_kroa200 =  results_kroa200[np.argmin([x[1] for x in results_kroa200 ])][1] 
max_result_kroa200 =  results_kroa200[np.argmax([x[1] for x in results_kroa200 ])][1] 
mean_result_kroa200 = sum([x[1] for x in results_kroa200])/ITER
 
min_result_time_kroa200 =  results_time_kroa200[np.argmin(results_time_kroa200)] 
max_result_time_kroa200 =  results_time_kroa200[np.argmax(results_time_kroa200)]
mean_result_time_kroa200 = sum(results_time_kroa200)/ITER


############################

min_result_krob200 =  results_krob200[np.argmin([x[1] for x in results_krob200 ])][1]
max_result_krob200 =  results_krob200[np.argmax([x[1] for x in results_krob200 ])][1] 
mean_result_krob200 = sum([x[1] for x in results_krob200])/ITER

min_result_time_krob200 =  results_time_krob200[np.argmin(results_time_krob200 )] 
max_result_time_krob200 =  results_time_krob200[np.argmax(results_time_krob200)] 
mean_result_time_krob200 = sum(results_time_krob200)/ITER


####
print("ODLEGŁOŚCI:")
print(f"kroa200: {mean_result_kroa200} ({min_result_kroa200}-{max_result_kroa200})")
print(f"krob200: {mean_result_krob200} ({min_result_krob200}-{max_result_krob200})")
print("CZAS:")
print(f"kroa200: {mean_result_time_kroa200} ({min_result_time_kroa200}-{max_result_time_kroa200})")
print(f"krob200: {mean_result_time_krob200} ({min_result_time_krob200}-{max_result_time_krob200})")


best_kroa200 =  results_kroa200[np.argmin([x[1] for x in results_kroa200 ])][0]
best_krob200 =  results_krob200[np.argmin([x[1] for x in results_krob200 ])][0]

plot_result(kroa200_instance,*best_kroa200,"kroa200 HAE")
plot_result(krob200_instance,*best_krob200," krob200 HAE")

print("KONIEC")