

def type_2_cost(type_str, fuels_dict ) : 
    conversion = {
        "gasfired" : "gas(euro/MWh)", 
        "turbojet" : "kerosine(euro/MWh)"  
    }

    return fuels_dict[conversion[type_str]] + ( 0.3 * fuels_dict["co2(euro/ton)"] if type_str == "gasfired" else 0)