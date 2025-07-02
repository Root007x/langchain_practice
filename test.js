const response = await fetch(
  "https://20e9-103-101-98-28.ngrok-free.app/chain/invoke",
  {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      input: {
        language: "bn",
        text: "How are you?",
      },
      config: {},
      kwargs: {
        additionalProp1: {}
      },
    }),
  }
);

const data = await response.json()
console.log(data.output)