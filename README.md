# RecommendationSystem

the predict funtion:
𝑦𝑝𝑟𝑒𝑑[𝑢,𝑖]=𝑏𝑖𝑎𝑠𝑔𝑙𝑜𝑏𝑎𝑙+𝑏𝑖𝑎𝑠𝑢𝑠𝑒𝑟[𝑢]+𝑏𝑖𝑎𝑠𝑖𝑡𝑒𝑚[𝑖]+<𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑢𝑠𝑒𝑟[𝑢], 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑖𝑡𝑒𝑚[𝑖] > 

we need minimized the loss (add regularize terms)
∑𝑢,𝑖| 𝑦𝑝𝑟𝑒𝑑[𝑢,𝑖] − 𝑦𝑡𝑟𝑢𝑒[𝑢,𝑖] |2 + 𝜆(|  𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑢𝑠𝑒𝑟[𝑢] | 2 + | 𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔𝑖𝑡𝑒𝑚[𝑖] | 2)
