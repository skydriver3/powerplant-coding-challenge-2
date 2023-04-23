from flask import Flask, request, jsonify
import numpy as np 
from Simplex import simplex 
from Helper import type_2_cost
from scipy.optimize import linprog 


app = Flask(__name__) 


@app.route('/productionplan', methods=['POST'])
def productionplan():

    payload = request.json
    
    pps = payload["powerplants"]
    pps_no_wind = [] 
    windturbines = [] 

    load = payload["load"]

    # Wind energy is free so we always want to use the windturbine to their max capacity
    for pp in pps : 
        if pp["type"] != "windturbine" : 
            pps_no_wind.append(pp)
        else  : 
            # Since we use the windturbine, we have to decrease the load that we require from the rest
            power =  pp["pmax"] * (payload["fuels"]["wind(%)"]  / 100) 
            windturbines.append((pp, power)) 
            load -= power

    n_pps = len(pps_no_wind)
    
    # Coefficient for the objective function
    c = np.array([type_2_cost( pp["type"], payload["fuels"] ) / pp["efficiency"] for pp in pps_no_wind ])

    # Constraint on the load 
    A_eq = np.array([np.ones(n_pps)]) 
    b_eq = np.array([load])

    # Constraints on Pmax and Pmin
    bounds = [(pp["pmin"], pp["pmax"]) for pp in pps_no_wind]


    # find solution
    res = simplex(c = c, A_eq = A_eq, b_eq = b_eq, bounds=bounds)

    # Test against linprog 
    res_x = linprog(c = c, A_eq=A_eq, b_eq = b_eq, bounds = bounds)["x"]
    print(f"custom : {res} \ncorrect : {res_x} ")


    # Return a response
    response = [
        {
            "name" : pps_no_wind[i]["name"], 
            "p" : res[i]
        }
        for i in range(n_pps)
    ] + [
        {
            "name" : pp["name"], 
            "p" : power
        }
        for pp, power in windturbines
    ]
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port = 8888, host="0.0.0.0")