const searchBox = document.getElementById("search");

if (searchBox) {
  searchBox.addEventListener("keypress", function (e) {
    if (e.key === "Enter") {
      const q = this.value.toLowerCase();
      if (q.includes("video")) window.location = "/";
      else if (q.includes("speech")) window.location = "/";
      else if (q.includes("image")) window.location = "/";
      else if (q.includes("chat")) window.location = "/chat";
    }
  });
}
