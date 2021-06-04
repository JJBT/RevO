function main(config, fn, target) {
  target = $(target);
  target.html(`
    <div class="box-main">
      <div class="before-wrapper">
        <div class="input_interfaces_query">

        </div>

        <div class="before__min-wrapper">
          <div class="input_interfaces_support">
          <!-- 3 дочерних div -->
          </div>
        </div>
      </div>
      <div class="output_interfaces">
      </div>

    </div>
    <div class="btn-wrapper">
      <button class="clear btn btn-clear">Clear</button>
      <button class="submit btn btn-submit">Submit</button>
    </div

  `);
  let io_master = Object.create(io_master_template);
  io_master.fn = fn
  io_master.target = target;
  io_master.config = config;

  let id_to_interface_map = {}

  function set_interface_id(interface, id) {
    interface.id = id;
    id_to_interface_map[id] = interface;
  }

  _id = 0;
  let input_interfaces = [];
  let output_interfaces = [];
  for (let i = 0; i < config["input_interfaces"].length; i++) {
    input_interface_data = config["input_interfaces"][i];
    input_interface = Object.create(image_input);

    if (input_interface_data[1]["name"] === 'query') {
      target.find(".input_interfaces_query").append(`
      <div class="input_interface interface" interface_id=${_id}>
        ${input_interface.html}
      </div>
      `);

    } else {
      target.find(".input_interfaces_support").append(`
      <div class="input_interface interface" interface_id=${_id}>
        ${input_interface.html}
      </div>
    `);
    }
    input_interface.target = target.find(`.input_interface[interface_id=${_id}]`);
    set_interface_id(input_interface, _id);
    input_interface.io_master = io_master;
    input_interface.init(input_interface_data[1]);
    input_interfaces.push(input_interface);
    _id++;
  }
  for (let i = 0; i < config["output_interfaces"].length; i++) {

    output_interface_data = config["output_interfaces"][i];
    output_interface = Object.create(image_output);

    target.find(".output_interfaces").append(`
      <div class="output_interface interface" interface_id=${_id}>
        ${output_interface.html}
      </div>
    `);

    output_interface.target = target.find(`.output_interface[interface_id=${_id}]`);
    set_interface_id(output_interface, _id);
    output_interface.io_master = io_master;
    output_interface.init(output_interface_data[1]);
    output_interfaces.push(output_interface);
    _id++;
  }
  io_master.input_interfaces = input_interfaces;
  io_master.output_interfaces = output_interfaces;

  function clear_all() {
    for (let input_interface of input_interfaces) {
      input_interface.clear();
    }
    for (let output_interface of output_interfaces) {
      output_interface.clear();
    }
  }  
  target.find(".clear").click(clear_all);

  target.find(".submit").show();
  target.find(".submit").click(function() {
    io_master.gather();
  })

  return io_master;
}
function main_url(config, url, target) {
  return main(config, function(data, action) {
    return new Promise((resolve, reject) => {
      $.ajax({type: "POST",
        url: url + action + "/",
        data: JSON.stringify({"data": data}),
        dataType: 'json',
        contentType: 'application/json; charset=utf-8',
        success: resolve,
        error: reject,
      });
    });              
  }, target);
}
