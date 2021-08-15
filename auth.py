from flask import Blueprint,render_template,request
import pickle
import numpy as np
auth = Blueprint('auth',__name__)
knn = pickle.load(open('knn.pkl','rb'))
svc = pickle.load(open('svc.pkl','rb'))
dtc = pickle.load(open('dtc.pkl','rb'))
rfc = pickle.load(open('rfc.pkl','rb'))

@auth.route('/dashboard')
def dashboard():
    return render_template('index.html')

@auth.route('/predictSVC')
def predictSVC():
    return render_template('predictSVC.html')
    
@auth.route('/predictKNN')
def predictKNN():
    return render_template('predictKNN.html')

@auth.route('/predictDTC')
def predictDTC():
    return render_template('predictDTC.html')
    
@auth.route('/predictRFC')
def predictRFC():
    return render_template('predictRFC.html')
    
@auth.route('/RFCA',methods=['POST'])
def rfca():
    req = request.form
    gender = int(req.get("gender"))
    age = int(req.get("age"))
    chol = int(req.get("chol"))
    press = int(req.get("press"))
    here = int(req.get("here"))
    smoke = int(req.get("smoke"))
    alco = int(req.get("alco"))
    phys = int(req.get("phys"))
    diab = int(req.get("diab"))
    diet = int(req.get("diet"))
    obes = int(req.get("obes"))
    stress = int(req.get("stress"))
    features = [gender,age,chol,press,here,smoke,alco,phys,diab,diet,obes,stress]
    final = [np.array(features)]
    predict = rfc.predict(final)
    output = predict.tolist()
    if(output[0] == 1):
        res = "Heart Disease Found!"
    else:
        res = "Heart Disease Not Found!"
    ans = [{
        "gender" : gender,    
        "age" : age,
        "chol" : chol,
        "press" : press,
        "here" : here,
        "smoke" : smoke,
        "alco" : alco,
        "phys" : phys,
        "diab" : diab,
        "diet" : diet,
        "obes" : obes,
        "stress" : stress,
        "res" : res,
    }]
    return render_template('rfcaOutput.html',ans=ans)
    
@auth.route('/DTCA',methods=['POST'])
def dtca():
    req = request.form
    gender = int(req.get("gender"))
    age = int(req.get("age"))
    chol = int(req.get("chol"))
    press = int(req.get("press"))
    here = int(req.get("here"))
    smoke = int(req.get("smoke"))
    alco = int(req.get("alco"))
    phys = int(req.get("phys"))
    diab = int(req.get("diab"))
    diet = int(req.get("diet"))
    obes = int(req.get("obes"))
    stress = int(req.get("stress"))
    features = [gender,age,chol,press,here,smoke,alco,phys,diab,diet,obes,stress]
    final = [np.array(features)]
    predict = dtc.predict(final)
    output = predict.tolist()
    if(output[0] == 1):
        res = "Heart Disease Found!"
    else:
        res = "Heart Disease Not Found!"
    ans = [{
        "gender" : gender,    
        "age" : age,
        "chol" : chol,
        "press" : press,
        "here" : here,
        "smoke" : smoke,
        "alco" : alco,
        "phys" : phys,
        "diab" : diab,
        "diet" : diet,
        "obes" : obes,
        "stress" : stress,
        "res" : res,
    }]
    return render_template('dtcaOutput.html',ans=ans)
    
@auth.route('/SVCA',methods=['POST'])
def svca():
    req = request.form
    gender = int(req.get("gender"))
    age = int(req.get("age"))
    chol = int(req.get("chol"))
    press = int(req.get("press"))
    here = int(req.get("here"))
    smoke = int(req.get("smoke"))
    alco = int(req.get("alco"))
    phys = int(req.get("phys"))
    diab = int(req.get("diab"))
    diet = int(req.get("diet"))
    obes = int(req.get("obes"))
    stress = int(req.get("stress"))
    features = [gender,age,chol,press,here,smoke,alco,phys,diab,diet,obes,stress]
    final = [np.array(features)]
    predict = svc.predict(final)
    output = predict.tolist()
    if(output[0] == 1):
        res = "Heart Disease Found!"
    else:
        res = "Heart Disease Not Found!"
    ans = [{
        "gender" : gender,    
        "age" : age,
        "chol" : chol,
        "press" : press,
        "here" : here,
        "smoke" : smoke,
        "alco" : alco,
        "phys" : phys,
        "diab" : diab,
        "diet" : diet,
        "obes" : obes,
        "stress" : stress,
        "res" : res,
    }]
    return render_template('svcaOutput.html',ans=ans)
    
@auth.route('/KNNA',methods=['POST'])
def knna():
    req = request.form
    gender = int(req.get("gender"))
    age = int(req.get("age"))
    chol = int(req.get("chol"))
    press = int(req.get("press"))
    here = int(req.get("here"))
    smoke = int(req.get("smoke"))
    alco = int(req.get("alco"))
    phys = int(req.get("phys"))
    diab = int(req.get("diab"))
    diet = int(req.get("diet"))
    obes = int(req.get("obes"))
    stress = int(req.get("stress"))
    features = [gender,age,chol,press,here,smoke,alco,phys,diab,diet,obes,stress]
    final = [np.array(features)]
    predict = knn.predict(final)
    output = predict.tolist()
    if(output[0] == 1):
        res = "Heart Disease Found!"
    else:
        res = "Heart Disease Not Found!"
    ans = [{
        "gender" : gender,    
        "age" : age,
        "chol" : chol,
        "press" : press,
        "here" : here,
        "smoke" : smoke,
        "alco" : alco,
        "phys" : phys,
        "diab" : diab,
        "diet" : diet,
        "obes" : obes,
        "stress" : stress,
        "res" : res,
    }]
    return render_template('knnaOutput.html',ans=ans)
