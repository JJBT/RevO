const image_input = {
  html: `
    <div class="sketch_tools hide">
      <div id="brush_1" size="8" class="brush selected"></div>
      <div id="brush_2" size="16" class="brush"></div>
      <div id="brush_3" size="24" class="brush"></div>
    </div>
    <div class="view_holders">
      <div class="canvas_holder">
        <canvas class="sketch"></canvas>
      </div>
    </div>        
    `,

  init: function (opts) {
    var io = this;
    this.shape = opts.shape;

    var dimension = Math.min(this.target.find(".canvas_holder").width(),
      this.target.find(".canvas_holder").height()) // dimension - border
    var id = this.id;
    this.sketchpad = new Sketchpad({
      element: '.interface[interface_id=' + id + '] .sketch',
      width: this.shape[0],
      height: this.shape[1]
    });
    this.sketchpad.penSize = this.target.find(".brush.selected").attr("size");
    this.canvas = this.target.find('.canvas_holder canvas')[0];
    this.context = this.canvas.getContext("2d");
    this.target.find(".brush").click(function (e) {
      io.target.find(".brush").removeClass("selected");
      $(this).addClass("selected");
      io.sketchpad.penSize = $(this).attr("size");
    })
    this.clear();

  },
  submit: function () {
    var dataURL = this.canvas.toDataURL("image/png");
    this.io_master.input(this.id, dataURL);

  },
  clear: function () {
    this.context.fillStyle = "#000000";
    this.context.fillRect(0, 0, this.context.canvas.width, this.context.
      canvas.height);

    this.target.find(".saliency_holder").addClass("hide");
  },

}