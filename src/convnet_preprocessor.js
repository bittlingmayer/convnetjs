 (function(global) {
  "use strict";
     
    var arrUnique = global.arrUnique;

    // looks at a column i of data and guesses what's in it
    // returns results of analysis: is column numeric? How many unique entries and what are they?
    function guessColumn(data, c) {
      var numeric = true;
      var vs = [];
      for(var i=0,n=data.length;i<n;i++) {
        var v = data[i][c];
        vs.push(v);
        if(isNaN(v)) numeric = false;
      }
      var u = arrUnique(vs);
      if(!numeric) {
        // if we have a non-numeric we will map it through uniques to an index
        return {numeric:numeric, num:u.length, uniques:u};
      } else {
        return {numeric:numeric, num:u.length};
      }
    }
    
    // returns arr (csv parse)
    // and colstats, which contains statistics about the columns of the input
    // parsing results will be sent to onSuccess(msg) repeatedly
    function importData(arr, onSuccess, onFailure) {

      // find number of datapoints
      var N = arr.length;
      var t = [];
      onSuccess("found " + N + " data points");
      if(N === 0) {
          onFailure('no data points found?');
          return;
      }
      
      // find dimensionality and enforce consistency
      var D = arr[0].length;
      for(var i=0;i<N;i++) {
        var d = arr[i].length;
        if(d !== D) {
            onFailure('data dimension not constant: line ' + i + ' has ' + d + ' entries.');
            return;
        }
      }
      onSuccess("data dimensionality is " + (D-1));
      
      // go through columns of data and figure out what they are
      var colstats = [];
      for(var i=0;i<D;i++) {
        var res = guessColumn(arr, i);
        colstats.push(res);
        if(D > 20 && i>3 && i < D-3) {
          if(i==4) {
            onSuccess("..."); // suppress output for too many columns
          }
        } else {
          onSuccess("column " + i + " looks " + (res.numeric ? "numeric" : "NOT numeric") + " and has " + res.num + " unique elements");
        }
      }

      return {arr: arr, colstats: colstats, N:N, D:D};
   }

  // process input mess into vols and labels
  function makeDataset(labelix, import_train_data, opt_import_test_data) {

    var arr = opt_import_test_data ? opt_import_test_data.arr : import_train_data.arr;
    if (opt_import_test_data) { console.log("Make dataset of test data."); }
    var colstats = import_train_data.colstats;
    var N = import_train_data.N;
    var D = import_train_data.D;

    var data = [];
    var labels = [];
    for(var i=0;i<N;i++) {
      var arri = arr[i];
      
      // create the input datapoint Vol()
      var p = arri.slice(0, D-1);
      var xarr = [];
      for(var j=0;j<D;j++) {
        if(j===labelix) continue; // skip!

        if(colstats[j].numeric) {
          xarr.push(parseFloat(arri[j]));
        } else {
          var u = colstats[j].uniques;
          var ix = u.indexOf(arri[j]); // turn into 1ofk encoding
          for(var q=0;q<u.length;q++) {
            if(q === ix) { xarr.push(1.0); }
            else { xarr.push(0.0); }
          }
        }
      }
      var x = new convnetjs.Vol(xarr);
      
      // process the label (last column)
      if(colstats[labelix].numeric) {
        var L = parseFloat(arri[labelix]); // regression
      } else {
        var L = colstats[labelix].uniques.indexOf(arri[labelix]); // classification
        if(L==-1) {
          console.log('whoa label not found! CRITICAL ERROR, very fishy.');
        }
      }
      data.push(x);
      labels.push(L);
    }
    
    var dataset = {};
    dataset.data = data;
    dataset.labels = labels;
    return dataset;
  }

  global.importData = importData;
  global.makeDataset = makeDataset;
  
})(convnetjs);
