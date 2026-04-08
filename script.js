async function loadData(){

const response = await fetch("../data/simulated_iot_data.csv");

const text = await response.text();

const rows = text.split("\n").slice(1);

let devices=[]
let packets=[]

const table=document.querySelector("#alertTable tbody")

rows.forEach(row=>{

const cols=row.split(",")

if(cols.length<3) return

const device=cols[0]
const packet=parseInt(cols[1])
const status=cols[3]

devices.push(device)
packets.push(packet)

if(status==="Suspicious"){

const tr=document.createElement("tr")

tr.innerHTML=`
<td>${device}</td>
<td>${packet}</td>
<td class="alert">${status}</td>
`

table.appendChild(tr)

}

})

createChart(devices,packets)

}

function createChart(devices,packets){

const ctx=document.getElementById("trafficChart")

new Chart(ctx,{

type:"bar",

data:{
labels:devices,
datasets:[{
label:"Packets",
data:packets
}]
}

})

}

loadData()
