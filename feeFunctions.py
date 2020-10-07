import xgboost as xgb
def feePrediction(pet):
    useful = pet[["AdoptionSpeed","MaturitySize","FurLength","Dewormed","Vaccinated","Sterilized","Health","Fee","Age"]]
    useful = pd.concat([useful, pd.get_dummies(train['Breed1'],prefix="Breed1"),pd.get_dummies(train['State'],prefix="State")], axis=1,join='inner')
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model('FeePredict')
    model = booster
    probs =  model.predict(xgb.DMatrix(data=useful))
    labels = [-1,0,1,2]
    fee_binned = []
    for line in probs:
        fee_binned.append(labels[np.argmax(line)])
        
    
    return fee_binned

def feeAdjustedPrediction(pet):
    useful = pet[["AdoptionSpeed","MaturitySize","FurLength","Dewormed","Vaccinated","Sterilized","Health","Fee","Age"]]
    useful = pd.concat([useful, pd.get_dummies(train['Breed1'],prefix="Breed1"),pd.get_dummies(train['State'],prefix="State")], axis=1,join='inner')
    clf = xgb.XGBClassifier()
    booster = xgb.Booster()
    booster.load_model('FeePredict2')
    model = booster
    probs =  model.predict(xgb.DMatrix(data=useful))
    labels = [-1,0,1,2]
    fee_binned = []
    for line in probs:
        fee_binned.append(labels[np.argmax(line)])
        
    
    return fee_binned

