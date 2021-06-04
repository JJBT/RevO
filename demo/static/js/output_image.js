const image_output = {
  html: `
    <div class="view_holders">
        <div class="saliency_holder hide">
          <canvas class="saliency"></canvas>
        </div>          
        <div class="output_image_holder hide">
          <img class="output_image">
        </div>
      </div>  
    `,
  init: function(opts) {},
  output: function(data) {
    this.target.find(".view_holder_parent").addClass("interface_box");
    this.target.find(".output_image_holder").removeClass("hide");
    this.target.find(".output_image").attr('src', data);
  },
  clear: function() {
    this.target.find(".view_holder_parent").removeClass("interface_box");
    this.target.find(".output_image_holder").addClass("hide");
    this.target.find(".saliency_holder").addClass("hide");
    this.target.find(".output_image").attr('src', "")
  },
}