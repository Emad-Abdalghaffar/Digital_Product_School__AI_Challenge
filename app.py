from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import keras


#_______________________________________________________________________
# Prediction function
# predicting multi-instances, compared with the actual labels. ## yet, handled-case in the notebook only.
# or predicting a single instance. ## handled-case in both: notebook & web-app.

def predict(records, model):
    
    # preparation
    
    # converting the input records to a dataframe, for such cases when the input is a dictionary (json) not an already-dataframe.
    # plus, resetting the indexes in order to be matchable & comparable to the later-constructed list of normalized records (norm_recs).
    # as in a case, the input maybe dataframe's slice of rows (records) starting with index != 0, like in the x_valid df.
    records= pd.DataFrame(records).reset_index().drop(columns=['index'])
    # casting the numeric input-features into "int64"
    records.iloc[:,2]= records.iloc[:,2].astype("int64")
    records.iloc[:,3]= records.iloc[:,3].astype("int64")
    
    # initializing an empty list, in order to contain the input (records) after being normalized.
    norm_recs= []
    
    # preprocessing
    for _index , record in records.iterrows():
        
        # initializing an empty list, that hosts the normalized features of each input reocrd.
        norm_rec= []
        
        # implementation of one-hot encoding & MinMax Scaling (normalization), instead of using the sklearn's. ####(for fun)####

        # one-hot encoding (nominal features)
        if record[0] == "Alkoholunfälle":
            norm_rec.extend([1,0,0])
        elif record[0] == "Fluchtunfälle":
            norm_rec.extend([0,1,0])
        elif record[0] == "Verkehrsunfälle":
            norm_rec.extend([0,0,1])

        if record[1] == "Verletzte und Getötete":
            norm_rec.extend([1,0,0])
        elif record[1] == "insgesamt":
            norm_rec.extend([0,1,0])
        elif record[1] == "mit Personenschäden":
            norm_rec.extend([0,0,1])

        # Normalization (numerical features) = (x - xmin) / (xmax - xmin)
        norm_rec.extend([(record[2]-2000)/20, (record[3]-1)/11])
    
        norm_recs.append(norm_rec)
    
    # printing the original and normalized records
    print("\n Original Records \n", records, "\n\n", "\n Normalized Records \n", pd.DataFrame(norm_recs))
    
    # printing the prediction
    
    # multi-instances, compared with the actual labels. (validation data slice)
    if records.shape[0] > 1:
        output= pd.concat([pd.DataFrame(model.predict(np.array(norm_recs)).astype(int), columns=["Prediction"]),
                               records["WERT"].rename('Actual', inplace = True).astype(int)],
                              axis=1)
        print("\n\n\n", "Prediction:", output)
        # display(output)

    # single instance (test datapoint)
    else:
        output= int(model.predict(np.array(norm_recs)).item())
        print("\n\n\n", "Prediction:\n", output, "\n")
  
    return output


#_______________________________________________________________________

app= Flask(__name__, template_folder='templates')

@app.route("/")
def landingPage():
    return render_template("web-app_page.html")

@app.route("/prediction", methods= ['POST'])
def prediction():

    datapoint= [x for x in request.form.values()]
    print(type(datapoint), datapoint)

    datapoint_to_ml= {"MONATSZAHL": datapoint[0],
                      "AUSPRAEGUNG": datapoint[1],
                      "JAHR": [datapoint[2]],
                      "MONAT": [datapoint[3]]}
    print(datapoint)
    if len(datapoint[4]) != 0:
        if datapoint[4] == "sq_model.h5":
            model = keras.models.load_model(datapoint[4])
        else:
            model = joblib.load(datapoint[4])
    else:
        model = joblib.load("MLPreg_model.pkl")
    
    numOfAcc= predict(datapoint_to_ml, model)

    return render_template("web-app_page.html",
                           prediction_preamble= "Prediction: {}✍🏻".format(abs(numOfAcc)))


if __name__ == "__main__":
    app.run(debug=True)