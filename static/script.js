$(document).ready(function() {

    var numOfBtn = 4;
    var flag = false;
    $('.switch').click(function() {
        //Без этого работает дважды
        if(flag){
            flag = false;
            return;
        }
        else{
            flag = true;
        }
        var num = $('input', this).attr('id');
        for(var i = 1; i <= numOfBtn; i++){
            if(num == 'switch-'+ i){
                ajax(i);
            }
            else{
                $('#switch-'+ i).prop("checked",false);
            }
        }
    });

    $("#nameValue").click(function () {
        var value = $("#nameValue").val();
        if(value == "yaadmin"){
            $("#trainbtn").prop("hidden", false);
            $("#changebtn").prop("hidden", false);
        }
    });

    $("#changebtn").click(function(){
        $.ajax
        (
            {
                type:'POST',
                url: '/photo',
                contentType: "application/json",
                dataType: 'json',
                data:JSON.stringify({'name': $("#nameValue").val()}),
                cache: false,
                async: false,
                success: function (response) {
                    if(response.result.admin == "true")
                    {
                        $("#passValue").prop("hidden", false);
                    }
                },
                error: function(request, status, error) {
                    var statusCode = request.status; // вот он код ответа
                    console.log(statusCode);
                }
            }
        );
    });

    $("#trainbtn").click(function(){
        //$("#changebtn").disable();
        $.ajax
        (
            {
                type:'POST',
                url: '/train',
                contentType: "application/json",
                dataType: 'json',
                data:JSON.stringify({'password': $("#passValue").val()}),
                cache: false,
                async: false,
                success: function (response) {
                    if(response.result.itisadmin == "true")
                    {
                        $("#adminbtn").prop("hidden", false);

                    }
                },
                error: function(request, status, error) {
                    var statusCode = request.status; // вот он код ответа
                    console.log(statusCode);
                }
            }
        );
    });

    $("#adminbtn").click(function(){
        //$("#changebtn").disable();
        $.ajax
        (
            {
                type:'POST',
                url: '/admin',
                contentType: "application/json",
                dataType: 'json',
                data:JSON.stringify({}),
                cache: false,
                async: false,
                success: function (response) {
                    if(response.response.reset == "true"){
                        $("#adminbtn").prop("hidden", true);
                        $("#passValue").prop("hidden", true);
                    }
                },
                error: function(request, status, error) {
                    var statusCode = request.status; // вот он код ответа
                    console.log(statusCode);
                }
            }
        );
    });
});