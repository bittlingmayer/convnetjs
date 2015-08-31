    var randperm = convnetjs.randperm;
    var importData = convnetjs.importData;
    var importData = convnetjs.importData;
    var makeDataset = convnetjs.makeDataset;

    function FAIL(outdivid, msg) {
      $(outdivid).prepend("<div class=\"msg\" style=\"background-color:#FCC;\">"+msg+"</div>")
    }
    function SUCC(outdivid, msg) {
      $(outdivid).prepend("<div class=\"msg\" style=\"background-color:#CFC;\">"+msg+"</div>")
    }


  // optionally provide a magic net
  function testEval(optional_net) {
    if (typeof optional_net !== 'undefined') {
      var net = optional_net;
    } else {
      var net = magicNet;
    }

    // set options for magic net
    net.ensemble_size = parseInt($("#ensemblenum").val())

    // read in the data in the text field
    var test_dataset = importTestData();
    // use magic net to predict
    var n = test_dataset.data.length;
    var acc = 0.0;
    for(var i=0;i<n;i++) {
      var yhat = net.predict(test_dataset.data[i]);
      if(yhat === -1) {
        $("#testresult").html("The MagicNet is not yet ready! It must process at least one batch of candidates across all folds first. Wait a bit.");
        $("#testresult").css('background-color', '#FCC');
        return;
      }
      var l = test_dataset.labels[i];
      acc += (yhat === l ? 1 : 0); // 0-1 loss
      console.log('test example ' + i + ': predicting ' + yhat + ', ground truth is ' + l);
    }
    acc /= n;

    // report accuracy
    $("#testresult").html("Test set accuracy: " + acc);
    $("#testresult").css('background-color', '#CFC');
  }

  function reinitGraph() {
    var legend = [];
    for(var i=0;i<magicNet.candidates.length;i++) {
      legend.push('model ' + i);
    }
    valGraph = new cnnvis.MultiGraph(legend, {miny: 0, maxy: 1});
  }

  var folds_evaluated = 0;
  function finishedFold() {
    folds_evaluated++;
    $("#foldreport").html("So far evaluated a total of " + folds_evaluated + "/" + magicNet.num_folds + " folds in current batch");
    reinitGraph();
  }
  var batches_evaluated = 0;
  function finishedBatch() {
    batches_evaluated++;
    $("#candsreport").html("So far evaluated a total of " + batches_evaluated + " batches of candidates");
  }

  var magicNet = null;
  function startCV() { // takes in train_dataset global
    var opts = {}
    opts.train_ratio = parseInt($("#trainp").val())/100.0;
    opts.num_folds = parseInt($("#foldsnum").val());
    opts.num_candidates = parseInt($("#candsnum").val());
    opts.num_epochs = parseInt($("#epochsnum").val());
    opts.neurons_min = parseInt($("#nnmin").val());
    opts.neurons_max = parseInt($("#nnmin").val());
    magicNet = new convnetjs.MagicNet(train_dataset.data, train_dataset.labels, opts);
    magicNet.onFinishFold(finishedFold);
    magicNet.onFinishBatch(finishedBatch);

    folds_evaluated = 0;
    batches_evaluated = 0;
    $("#candsreport").html("So far evaluated a total of " + batches_evaluated + " batches of candidates");
    $("#foldreport").html("So far evaluated a total of " + folds_evaluated + "/" + magicNet.num_folds + " folds in current batch");
    reinitGraph();

    var legend = [];
    for(var i=0;i<magicNet.candidates.length;i++) {
      legend.push('model ' + i);
    }
    valGraph = new cnnvis.MultiGraph(legend, {miny: 0, maxy: 1});
    setInterval(step, 0);
  }
      
    var fold;
    var cands = [];
    var dostep = false;
    var valGraph;
    var iter = 0;
    function step() {
      iter++;
      
      magicNet.step();
      if(iter % 300 == 0) {

        var vals = magicNet.evalValErrors();
        valGraph.add(magicNet.iter, vals);
        valGraph.drawSelf(document.getElementById("valgraph"));
    
        // print out the best models so far
        var cands = magicNet.candidates; // naughty: get pointer to internal data
        var scores = [];
        for(var k=0;k<cands.length;k++) {
          var c = cands[k];
          var s = c.acc.length === 0 ? 0 : c.accv / c.acc.length;
          scores.push(s);
        }
        var mm = convnetjs.maxmin(scores);
        var cm = cands[mm.maxi];
        var t = '';
        if(c.acc.length > 0) {
          t += 'Results based on ' + c.acc.length + ' folds:';
          t += 'best model in current batch (validation accuracy ' + mm.maxv + '):<br>';
          t += '<b>Net layer definitions:</b><br>';
          t += JSON.stringify(cm.layer_defs);
          t += '<br><b>Trainer definition:</b><br>';
          t += JSON.stringify(cm.trainer_def);
          t += '<br>';
        }
        $('#bestmodel').html(t);

        // also print out the best model so far
        var t = '';
        if(magicNet.evaluated_candidates.length > 0) {
          var cm = magicNet.evaluated_candidates[0];
          t += 'validation accuracy of best model so far, overall: ' + cm.accv / cm.acc.length + '<br>';
          t += '<b>Net layer definitions:</b><br>';
          t += JSON.stringify(cm.layer_defs);
          t += '<br><b>Trainer definition:</b><br>';
          t += JSON.stringify(cm.trainer_def);
          t += '<br>';
        }
        $('#bestmodeloverall').html(t);
      }
    }

    var import_train_data, labelix, train_dataset; // globals
    function importTrainData() {
      var csv_txt = $('#data-ta').val();
      var arr = $.csv.toArrays(csv_txt);
      var arr_train = arr;
      var arr_test = [];

      var test_ratio = Math.floor($("#testsplit").val());
      if(test_ratio !== 0) {
        // send some lines to test set
        var test_lines_num = Math.floor(arr.length * test_ratio / 100.0);
        var rp = randperm(arr.length);
        arr_train = [];
        for(var i=0;i<arr.length;i++) {
          if(i<test_lines_num) {
            arr_test.push(arr[rp[i]]);
          } else {
            arr_train.push(arr[rp[i]]);
          }
        }
        // enter test lines to test box
        var t = "";
        for(var i=0;i<arr_test.length;i++) {
          t+= arr_test[i].join(",")+"\n";
        }
        $("#data-te").val(t);
        $("#datamsgtest").empty();
      }

      $("#prepromsg").empty(); // flush
      SUCC("#prepromsg", "Sent " + arr_test.length + " data to test, keeping " + arr_train.length + " for train.");
      var onSuccess = function (msg) { SUCC('#datamsg', msg)};
      var onFailure = function (msg) { FAIL('#datamsg', msg)};
      import_train_data = importData(arr_train, onSuccess, onFailure);
      labelix = parseInt($("#labelix").val());
      if(labelix < 0) labelix = import_train_data.D + labelix; // eg -1 should turn to D-1        
      train_dataset = makeDataset(labelix, import_train_data);
      return train_dataset;
    }

    function importTestData() {
      var csv_txt = $('#data-te').val();
      var arr = $.csv.toArrays(csv_txt);
      var onSuccess = function (msg) { SUCC('#datamsgtest', msg)};
      var onFailure = function (msg) { FAIL('#datamsgtest', msg)};
      var import_test_data = importData(arr, onSuccess, onFailure);
      // note important that we use colstats of train data!
      console.log("import_test_data.arr: " + import_test_data.arr);
      console.log("import_test_data.arr[0]: " + import_test_data.arr[0]);
      test_dataset = makeDataset(labelix, import_train_data, import_test_data);
      return test_dataset;
    }

    function loadDB(url) {
      // load a dataset from a url with ajax
      $.ajax({
        url: url,
        dataType: "text",
        success: function(txt) {
          $("#data-ta").val(txt);
        }
      });
    }

    function start() {
      loadDB('data/car.data.txt');
    }

    function exportMagicNet() {
      $("#taexport").val(JSON.stringify(magicNet.toJSON()));

      /*
      // for debugging
      var j = JSON.parse($("#taexport").val());
      var m = new convnetjs.MagicNet();
      m.fromJSON(j);
      testEval(m);
      */
    }

    function changeNNRange() {
      magicNet.neurons_min = parseInt($("#nnmin").val());
      magicNet.neurons_max = parseInt($("#nnmax").val());
    }
