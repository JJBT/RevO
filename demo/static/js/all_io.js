var io_master_template = {
  gather: function() {
    this.clear();
    for (let iface of this.input_interfaces) {
      iface.submit();
    }
  },
  clear: function() {
    this.last_input = new Array(this.input_interfaces.length);
    this.input_count = 0;
    if (this.gather_timeout) {
      window.clearTimeout(this.gather_timeout);
    }
  },
  input: function(interface_id, data) {
    this.last_input[interface_id] = data;
    this.input_count += 1;
    if (this.input_count == this.input_interfaces.length) {
      this.submit();
    }
  },
  submit: function() {
    let io = this;

    this.target.find(".output_interfaces").css("opacity", 0.5);

    this.fn(this.last_input, "predict").then((output) => {
      io.output(output);
    }).catch((error) => {
      console.error(error);
    });
  },

  output: function(data) {

    for (let i = 0; i < this.output_interfaces.length; i++) {
      this.output_interfaces[i].output(data["data"][i]);
    }
    this.target.find(".output_interfaces").css("opacity", 1);

  }
};


